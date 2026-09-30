import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None, *, timeout_s: float | None = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._timeout_s = timeout_s
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        deadline = None if self._timeout_s is None else time.monotonic() + self._timeout_s
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers,
                    open_timeout=self._timeout_s or 10,
                )
                try:
                    metadata = msgpack_numpy.unpackb(conn.recv(timeout=self._timeout_s))
                except Exception:
                    conn.close()
                    raise
                return conn, metadata
            except ConnectionRefusedError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Policy server connection deadline expired")
                logging.info("Still waiting for server...")
                time.sleep(5 if deadline is None else min(.1, max(0., deadline - time.monotonic())))

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        try:
            response = self._ws.recv(timeout=self._timeout_s)
        except TimeoutError:
            self._ws.close()
            raise
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
