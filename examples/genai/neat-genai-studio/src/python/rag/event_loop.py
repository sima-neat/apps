"""An asyncio event loop for the calling thread.

pymilvus's AsyncMilvusClient, which langchain-milvus creates even for sync
calls, asks for the current thread's event loop. Flask serves each request on
a worker thread that has none, so building a database there (the Studio's
upload and reset routes) fails with "There is no current event loop in
thread ..." unless one is set first. setup.sh runs on the main thread and
never hit this.
"""

import asyncio
import threading

# The loop ensure_thread_event_loop created for each thread, so the route that
# built a database can close it again (worker threads are reused).
_created = threading.local()


def ensure_thread_event_loop() -> asyncio.AbstractEventLoop:
    """Return this thread's event loop, creating and setting one if needed."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _created.loop = loop
        return loop


def close_thread_event_loop() -> None:
    """Close and unset the loop ensure_thread_event_loop created for this
    thread, if it created one; a loop the thread already had is left alone."""
    loop = getattr(_created, "loop", None)
    if loop is None:
        return
    _created.loop = None
    asyncio.set_event_loop(None)
    if not loop.is_closed():
        loop.close()
