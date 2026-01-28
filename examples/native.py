#!/usr/bin/env python
# coding: utf-8
# ruff: noqa

# # ArbiterOS Native Backend Tutorial
#
# This notebook demonstrates how to build a policy-governed agent using ArbiterOS's native backend.
#
# ## What You'll Learn
#
# - ✅ How to set up multiple tools with different schemas
# - ✅ How to enforce policies (e.g., must use tools before responding)
# - ✅ How to evaluate execution quality with custom evaluators
# - ✅ How to build an agent loop with dynamic tool selection
# - ✅ How policy violations are detected and reported
#
# ## Scenario
#
# We'll build an agent that:
# 1. Takes a user query: "What's 25 + 17? Also, what's the weather in San Francisco?"
# 2. Uses a **calculator tool** to compute the math
# 3. Uses a **weather tool** to fetch weather data
# 4. Generates a final response combining both results
#
# We'll also demonstrate what happens when the agent violates the policy by trying to respond without using tools!

# ## ⚙️ Environment Setup
#
# **Run the cells below to:**
# 1. Add ArbiterOS to Python's import path
# 2. Diagnose your environment and check if packages are installed

# In[1]:


# Add project root to Python path
import sys
from pathlib import Path

# Get the project root (parent of examples/)
project_root = Path.cwd().parent if Path.cwd().name == "examples" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"✅ Added to sys.path: {project_root}")

# Verify installation
try:
    import arbiteros_alpha

    print("✅ ArbiterOS-alpha imported successfully!")
    print(f"   Location: {arbiteros_alpha.__file__}")
except ImportError as e:
    print(f"❌ Failed to import arbiteros_alpha: {e}")
    print(f"   Current sys.path: {sys.path[:3]}")
    print("   Make sure you're running from the project directory!")


# ## 1. Setup and Imports

# In[2]:


import logging
from typing import Literal

from pydantic import BaseModel
from rich.logging import RichHandler

import arbiteros_alpha.instructions as Instr
from arbiteros_alpha import ArbiterOSAlpha
from arbiteros_alpha.evaluation import EvaluationResult, NodeEvaluator
from arbiteros_alpha.history import History
from arbiteros_alpha.policy import MustUseToolsChecker

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler()],
)

print("✅ Imports successful!")


# ## 2. Initialize ArbiterOS
#
# The `native` backend is the simplest way to use ArbiterOS - it doesn't require LangGraph.

# In[3]:


arbiter_os = ArbiterOSAlpha(backend="native")
print("✅ ArbiterOS initialized with native backend")


# ## 3. Define Schemas
#
# Each tool has its own input/output schema for type safety. This ensures that:
# - Tools receive the correct parameters
# - Outputs are validated before being used
# - Errors are caught early in development

# In[4]:


# Calculator tool schemas
class CalculatorInput(BaseModel):
    expression: str


class CalculatorOutput(BaseModel):
    result: str


# Weather tool schemas
class WeatherInput(BaseModel):
    city: str


class WeatherOutput(BaseModel):
    temperature: str
    condition: str
    status: str


# Agent decision schemas
class GenerateInput(BaseModel):
    query: str
    step: int
    skip_tools: bool = False


class GenerateOutput(BaseModel):
    action: Literal["use_calculator", "use_weather", "respond"]
    tool_input: dict | None = None


# Final response schemas
class RespondInput(BaseModel):
    calc_result: str
    weather_info: str


class RespondOutput(BaseModel):
    response: str


print("✅ All schemas defined")


# ## 4. Configure Policies
#
# **MustUseToolsChecker** prevents the agent from hallucinating answers without consulting tools.
#
# This is a common safety pattern:
# - ❌ Block responses without tool usage
# - ✅ Allow responses only after tools have been called

# In[5]:


arbiter_os.add_policy_checker(
    MustUseToolsChecker(
        name="must_use_tools_before_respond",
        respond_instruction=Instr.RESPOND,
    )
)

print("✅ Policy checker added: MustUseToolsChecker")


# ## 5. Configure Evaluators
#
# Evaluators assess the quality of execution **after** each instruction completes.
#
# Unlike PolicyCheckers (which can block execution), evaluators provide feedback scores.

# In[6]:


class ToolResultEvaluator(NodeEvaluator):
    """Evaluate whether tool calls return valid results."""

    def __init__(self):
        super().__init__(
            name="tool_result_validator", target_instructions=[Instr.TOOL_CALL]
        )

    def evaluate(self, history: History) -> EvaluationResult:
        current = history.entries[-1][-1]
        result = current.output_state.get("result") or current.output_state

        is_error = "error" in str(result).lower()
        is_empty = not result or result == ""

        score = 0.0 if (is_error or is_empty) else 1.0
        passed = score > 0.5
        feedback = "Valid tool result" if passed else "Invalid or empty result"

        return EvaluationResult(
            score=score,
            passed=passed,
            feedback=feedback,
            metadata={"result": str(result)},
        )


class ResponseCompletenessEvaluator(NodeEvaluator):
    """Evaluate whether final response includes all tool results."""

    def __init__(self):
        super().__init__(
            name="response_completeness", target_instructions=[Instr.RESPOND]
        )

    def evaluate(self, history: History) -> EvaluationResult:
        current = history.entries[-1][-1]
        response = current.output_state.get("response", "")

        has_calc = any(word in response for word in ["42", "calculation", "result"])
        has_weather = any(word in response for word in ["weather", "Sunny", "72"])

        score = (0.5 if has_calc else 0) + (0.5 if has_weather else 0)
        passed = score >= 0.8
        feedback = f"Response completeness: calc={has_calc}, weather={has_weather}"

        return EvaluationResult(
            score=score,
            passed=passed,
            feedback=feedback,
            metadata={"has_calc": has_calc, "has_weather": has_weather},
        )


arbiter_os.add_evaluator(ToolResultEvaluator())
arbiter_os.add_evaluator(ResponseCompletenessEvaluator())

print("✅ Evaluators added: ToolResultEvaluator, ResponseCompletenessEvaluator")


# ## 6. Define Tool Functions
#
# Each tool is decorated with `@arbiter_os.instruction()` and has its own schema.
#
# ArbiterOS automatically validates inputs/outputs against these schemas.

# In[7]:


@arbiter_os.instruction(
    Instr.TOOL_CALL, input_schema=CalculatorInput, output_schema=CalculatorOutput
)
def calculator(expression: str) -> dict:
    """Calculate a math expression."""
    print(f"🔢 Calculator: {expression}")
    result = str(eval(expression))  # Simple eval for demo
    print(f"   Result: {result}")
    return {"result": result}


@arbiter_os.instruction(
    Instr.TOOL_CALL, input_schema=WeatherInput, output_schema=WeatherOutput
)
def get_weather(city: str) -> dict:
    """Get weather for a city."""
    print(f"🌤️  Weather API: {city} is failed!")
    # Simulated weather data
    weather_data = {"temperature": "", "condition": "", "status": "error"}
    print(f"{weather_data=}")
    return weather_data


# Tool registry for dynamic dispatch
TOOLS = {
    "calculator": calculator,
    "weather": get_weather,
}

print("✅ Tools defined: calculator, get_weather")


# ## 7. Define Agent Decision Function
#
# The `generate` function decides what the agent should do next.
#
# In this simple example, it's deterministic based on step number:
# - Step 0: Use calculator
# - Step 1: Use weather tool
# - Step 2+: Generate response

# In[8]:


@arbiter_os.instruction(
    Instr.GENERATE, input_schema=GenerateInput, output_schema=GenerateOutput
)
def generate(query: str, step: int, skip_tools: bool = False) -> dict:
    """Deterministic agent that decides next action based on step number."""
    print(f"\n🤖 Agent thinking (step {step})...")

    # BAD PATH: Skip tools and go directly to respond (violates policy!)
    if skip_tools:
        print("   ⚠️  [POLICY VIOLATION] Skipping tools, going directly to respond!")
        return {"action": "respond", "tool_input": None}

    # GOOD PATH: Normal execution
    if step == 0:
        print("   → Need to calculate 25 + 17")
        return {
            "action": "use_calculator",
            "tool_input": {"expression": "25 + 17"},
        }
    elif step == 1:
        print("   → Need to check weather in San Francisco")
        return {"action": "use_weather", "tool_input": {"city": "San Francisco"}}
    else:
        print("   → Ready to respond")
        return {"action": "respond", "tool_input": None}


@arbiter_os.instruction(
    Instr.RESPOND, input_schema=RespondInput, output_schema=RespondOutput
)
def respond(calc_result: str, weather_info: str) -> dict:
    """Generate final response combining all tool results."""
    response = (
        f"The calculation result is {calc_result}. "
        f"The weather in San Francisco is {weather_info}."
    )
    print(f"\n💬 Final Answer: {response}")
    return {"response": response}


print("✅ Agent functions defined: generate, respond")


# ## 8. Define Main Agent Loop
#
# The agent loop:
# 1. Calls `generate()` to decide next action
# 2. Dispatches to the appropriate tool
# 3. Repeats until the agent decides to respond

# In[ ]:


@arbiter_os.rollout()
def run_agent(skip_tools: bool = False):
    """Run agent loop that dynamically calls tools based on decisions."""
    user_query = "What's 25 + 17? Also, what's the weather in San Francisco?"

    scenario = "❌ POLICY VIOLATION SCENARIO" if skip_tools else "✅ NORMAL SCENARIO"
    print("=" * 70)
    print(f"{scenario}")
    print(f"🎯 User Query: {user_query}")
    print("=" * 70)

    # Store results from tool calls
    tool_results = {}
    step = 0
    max_iterations = 10

    # Agent loop
    for iteration in range(max_iterations):
        # Agent decides next action
        decision = generate(query=user_query, step=step, skip_tools=skip_tools)
        action = decision["action"]

        if action == "use_calculator":
            result = TOOLS["calculator"](**decision["tool_input"])
            tool_results["calc_result"] = result["result"]
            step += 1

        elif action == "use_weather":
            result = TOOLS["weather"](**decision["tool_input"])
            tool_results["weather_info"] = (
                f"{result['condition']}, {result['temperature']}"
            )
            step += 1

        elif action == "respond":
            # Provide dummy values if tools were skipped
            if not tool_results:
                tool_results = {"calc_result": "UNKNOWN", "weather_info": "UNKNOWN"}

            final_response = respond(**tool_results)
            print(final_response["response"])
            print("\n" + "=" * 70)
            print(f"✅ Task Completed in {iteration + 1} iterations")
            print("=" * 70)
            break
        else:
            print(f"⚠️  Unknown action: {action}")
            break
    else:
        print(f"\n⚠️  Max iterations ({max_iterations}) reached")


print("✅ Agent loop defined")


# ## 9. Run Normal Execution (Success)
#
# Let's run the agent in normal mode. It should:
# 1. Use the calculator tool
# 2. Use the weather tool
# 3. Generate a response
#
# **All policies should pass ✅**

# In[10]:


print("EXECUTION 1: NORMAL FLOW")

run_agent(skip_tools=False)


# ### View Execution History (Normal Flow)
#
# The history shows:
# - All instructions executed
# - Policy checks (all passed)
# - Evaluator scores (should be 1.0)

# In[11]:


arbiter_os.history.pprint()


# ## 10. Run Policy Violation Scenario
#
# Now let's see what happens when the agent violates the policy!
#
# We'll set `skip_tools=True`, causing the agent to:
# 1. Skip all tool calls
# 2. Try to respond immediately
#
# **Expected: MustUseToolsChecker should detect the violation ❌**

# In[12]:


print("EXECUTION 2: POLICY VIOLATION FLOW")
run_agent(skip_tools=True)


# ### View Execution History (Violation Flow)
#
# Notice in the history:
# - ❌ Policy check failed for RESPOND instruction
# - The agent still completed (non-blocking mode)
# - Response contains "UNKNOWN" values (no tool results)

# In[13]:


arbiter_os.history.pprint()


# ## Summary
#
# In this tutorial, you learned:
#
# ### ✅ Key Concepts
#
# 1. **Schema Validation**: Use Pydantic models to validate inputs/outputs
# 2. **Policy Enforcement**: `MustUseToolsChecker` prevents responses without tool usage
# 3. **Evaluation**: Assess execution quality with custom evaluators
# 4. **Agent Loops**: Build dynamic agents that choose tools based on context
# 5. **History Tracking**: All executions are logged with policy checks and evaluations
#
# ### 🔧 Try It Yourself
#
# Modify the code to:
# - Add a new tool (e.g., currency converter)
# - Create a custom policy checker
# - Change the evaluator thresholds
# - Make the agent decision-making more sophisticated (use an LLM!)
#
# ### 📚 Next Steps
#
# - Explore LangGraph backend for graph-based workflows
# - Learn about PolicyRouters for dynamic flow control
# - Check out verification requirements for high-stakes operations
