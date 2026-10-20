# Tracing

The agent exports OpenTelemetry traces over OTLP to Jaeger when
`OTEL_EXPORTER_OTLP_ENDPOINT` is set. Open the Jaeger UI at http://localhost:16686 to
browse them.

Each task is one trace. The root span, `agent.task`, records the number of steps, the total
cost and the total tokens. Under it there is an `llm.call` span for each model call, with
the model, input tokens, output tokens and cost, and a `tool.call` span for each tool call,
with the tool name, the policy decision, the duration in milliseconds and whether it
succeeded.
