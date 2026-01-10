from flask import Flask, Response, render_template_string  # type: ignore

from ...utils import get_curdir
from ..improc import jpgencode
from .camera import IpcamCapture

__all__ = ["WebDemo"]


def gen(cap, pipelines=None):
    if pipelines is None:
        pipelines = []
    while True:
        frame = cap.get_frame()
        for f in pipelines:
            frame = f(frame)
        frame_bytes = jpgencode(frame)
        if frame_bytes is None:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n"
        )


class WebDemo:
    def __init__(
        self,
        camera_ip: str,
        color_base: str = "BGR",
        route: str = "/",
        pipelines: list | None = None,
    ):
        if pipelines is None:
            pipelines = []
        app = Flask(__name__)

        @app.route(route)
        def _index():
            with open(
                str(get_curdir(__file__) / "video_streaming.html"),
                encoding="utf-8",
            ) as file:
                html_content = file.read()
            return render_template_string(html_content)

        @app.route("/video_feed")
        def _video_feed():
            return Response(
                gen(
                    cap=IpcamCapture(camera_ip, color_base), pipelines=pipelines
                ),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        self.app = app

    def run(self, host="0.0.0.0", port=5001, debug=False, threaded=True):
        self.app.run(host=host, port=port, debug=debug, threaded=threaded)
