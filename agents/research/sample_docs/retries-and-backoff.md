# Retries and backoff

Calls to a model API fail for reasons that have nothing to do with the request: rate limits
(HTTP 429), overloaded servers (529), timeouts and dropped connections. These are transient
and worth retrying. Bad requests (400) and authentication errors (401) are not.

The runtime retries transient errors with exponential backoff: the delay doubles after each
failed attempt (1s, 2s, 4s, ...) up to a maximum delay of 30 seconds. Full jitter picks a
random delay between zero and that value, so many workers that failed together do not all
retry at the same moment.

Each attempt is traced as its own `llm.call` span, so failed attempts are visible in Jaeger.
