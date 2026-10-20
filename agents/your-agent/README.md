# Your agent

A starting point for your own agent. The research agent in `agents/research/` is a complete,
working example; the quickest way to build yours is to copy it and change three things: the
tools, the prompt and the report.

## Steps

1. **Copy the research agent under a valid Python package name** (underscores, not hyphens):

   ```bash
   cp -r agents/research agents/support_bot
   ```

   Then replace `agents.research` with `agents.support_bot` in the imports, and give it its
   own `QUEUE_NAME` in `agent.py`.

2. **Replace the tools** in `tools/`. Each tool is a class with a `name`, a `description`,
   an `input_schema` (JSON Schema) and an async `run(args) -> str`. Register them in
   `build_registry()`. Raise an exception for bad input or failures; the runtime reports it
   to the model as a tool error instead of failing the task.

3. **Add a policy rule for every tool** in `policies/default.yaml` (or a policy file of your
   own, selected with `AGENT_POLICY_FILE`). Tools without a rule are denied. Use
   `require_approval` for anything with side effects, and `allowed_paths` or
   `allowed_domains` to limit file and network access.

4. **Rewrite `prompt.md`.** Describe the job, when to use each tool, and the exact format of
   the final answer.

5. **Shape the result.** Replace `ResearchReport` and `parse_final_answer()` with whatever
   your callers need. `TaskResult` gives you the final text, the tools called, the number of
   LLM calls and the total cost.

6. **Test it with the fake LLM client.** `tests/agents/research/test_research_agent.py`
   shows how to script the model's responses and run the whole agent, tools included,
   without an API key.

7. **Run it:**

   ```bash
   python -m agents.support_bot --sync "a question"     # in-process
   python -m agents.support_bot "a question"            # enqueue
   python -m core.queue.worker --queue support_bot --handler agents.support_bot.agent:handle_job
   ```

`agent.py` in this folder is a skeleton with the same structure, marked with TODOs. This
folder is a template, not a package: its name has a hyphen, so it cannot be imported until you
copy it to a name with underscores.
