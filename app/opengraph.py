"""Link previews (Discord, Slack, X...): replace Gradio's default tags with the project's own.

Gradio hard-codes "Gradio / Click to try out the app!" Open Graph tags in its page
template and offers no option to remove them, so the home page HTML is rewritten
on the way out. Every other route (API, streaming, files) is passed through untouched.
"""
import html
import re

from app.settings import DESCRIPTION, PUBLIC_URL, TITLE

OG_IMAGE_PATH = "/og-image.png"

_PREVIEW_TAG = re.compile(r'<meta\s[^>]*?(?:property|name)="(?:og|twitter):[^>]*>\s*', re.IGNORECASE)


def preview_tags() -> str:
    title, description = html.escape(TITLE), html.escape(DESCRIPTION)
    image = f"{PUBLIC_URL}{OG_IMAGE_PATH}"
    return f"""
    <meta name="description" content="{description}" />
    <meta property="og:type" content="website" />
    <meta property="og:url" content="{PUBLIC_URL}/" />
    <meta property="og:title" content="{title}" />
    <meta property="og:description" content="{description}" />
    <meta property="og:image" content="{image}" />
    <meta property="og:image:width" content="1200" />
    <meta property="og:image:height" content="630" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="{title}" />
    <meta name="twitter:description" content="{description}" />
    <meta name="twitter:image" content="{image}" />
    <meta name="theme-color" content="#111111" />
"""


def rewrite_head(page: str) -> str:
    page = _PREVIEW_TAG.sub("", page)
    return page.replace("</head>", preview_tags() + "</head>", 1)


class OpenGraphMiddleware:
    """ASGI middleware rewriting the preview tags of the home page only."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/" or scope["method"] != "GET":
            await self.app(scope, receive, send)
            return

        start, chunks = {}, []

        async def rewrite_send(message):
            if message["type"] == "http.response.start":
                start.update(message)
                return
            if message["type"] != "http.response.body":
                await send(message)
                return
            chunks.append(message.get("body", b""))
            if message.get("more_body"):
                return

            body = b"".join(chunks)
            headers = dict(start.get("headers", []))
            is_html = headers.get(b"content-type", b"").startswith(b"text/html")
            if is_html and b"content-encoding" not in headers:
                body = rewrite_head(body.decode("utf-8")).encode("utf-8")
            headers[b"content-length"] = str(len(body)).encode()
            await send({**start, "headers": list(headers.items())})
            await send({"type": "http.response.body", "body": body})

        await self.app(scope, receive, rewrite_send)
