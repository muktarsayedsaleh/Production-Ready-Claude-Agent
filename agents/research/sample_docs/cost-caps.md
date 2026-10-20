# Cost caps

Every task has a cost cap in US dollars (default $1.00). After each model call the runtime
adds that call's cost, computed from the token counts and the pricing table, to the task's
running total. When the total goes over the cap, the task stops with a `CostCapExceeded`
error that states how much was spent and after how many calls.

Pricing is per million tokens. For example, at $4 per million input tokens and $20 per
million output tokens, a call with 1,000 input tokens and 500 output tokens costs
1,000 x 4 / 1,000,000 + 500 x 20 / 1,000,000 = $0.014.

The cap is checked after a call completes, so a task can overshoot it by at most one call.
