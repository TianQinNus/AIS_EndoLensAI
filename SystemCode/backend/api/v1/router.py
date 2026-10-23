"""Route registration boundary for the versioned API.

The current compatibility application still owns closure-scoped route handlers.
This module is the stable boundary for the next extraction into individual
routers without changing public paths.
"""

API_PREFIX = "/api/v1"
