"""Run the serving API: python -m serving."""

import uvicorn

from serving.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "serving.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
