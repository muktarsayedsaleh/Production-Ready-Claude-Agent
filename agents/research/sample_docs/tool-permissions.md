# Tool permissions

Before a tool runs, the runtime checks it against the policy in `policies/default.yaml`.
Each tool is `allow`, `deny` or `require_approval`, and can carry limits:

- `max_calls_per_task`: further calls are denied once the limit is reached.
- `allowed_paths`: path arguments must resolve inside one of these directories.
- `allowed_domains`: URL arguments must be on one of these domains or a subdomain.

A denied call is not executed. The model receives a structured error explaining why, and
can try something else. A call that requires approval pauses the job with the status
`waiting_approval` until a reviewer approves or rejects it, and then the job resumes from
where it stopped.
