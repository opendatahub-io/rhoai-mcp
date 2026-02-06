"""Port-forward manager for accessing Kubernetes services.

This module provides functionality to create and manage `oc port-forward` or
`kubectl port-forward` connections to Kubernetes services when running outside
the cluster. This allows direct access to internal services without requiring
external Routes or OAuth authentication.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import socket
from dataclasses import dataclass, field
from typing import ClassVar

logger = logging.getLogger(__name__)


class PortForwardError(Exception):
    """Error during port-forward operation."""

    pass


@dataclass
class PortForwardConnection:
    """Represents an active port-forward connection."""

    namespace: str
    service_name: str
    remote_port: int
    local_port: int
    process: asyncio.subprocess.Process
    ref_count: int = field(default=1)
    use_https: bool = field(default=False)

    @property
    def local_url(self) -> str:
        """Get the local URL for this port-forward."""
        protocol = "https" if self.use_https else "http"
        return f"{protocol}://localhost:{self.local_port}"

    def __hash__(self) -> int:
        return hash((self.namespace, self.service_name, self.remote_port))


class PortForwardManager:
    """Manages port-forward connections to Kubernetes services.

    This is a singleton class that maintains port-forward subprocesses
    with reference counting to share connections across multiple clients.

    Usage:
        manager = PortForwardManager.get_instance()
        conn = await manager.forward("my-namespace", "my-service", 8080)
        # Use conn.local_url to access the service
        await manager.release(conn)
    """

    _instance: ClassVar[PortForwardManager | None] = None
    _lock: ClassVar[asyncio.Lock | None] = None

    def __init__(self) -> None:
        self._connections: dict[tuple[str, str, int], PortForwardConnection] = {}
        self._cli_path: str | None = None

    @classmethod
    def get_instance(cls) -> PortForwardManager:
        """Get the singleton instance of PortForwardManager."""
        if cls._instance is None:
            cls._instance = PortForwardManager()
        return cls._instance

    @classmethod
    async def _get_lock(cls) -> asyncio.Lock:
        """Get the async lock, creating it if needed."""
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    def _find_free_port(self) -> int:
        """Find an available local port.

        Returns:
            An available port number.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port: int = s.getsockname()[1]
            return port

    def _find_oc_or_kubectl(self) -> str:
        """Find the oc or kubectl CLI in PATH.

        Returns:
            Path to the CLI executable.

        Raises:
            PortForwardError: If neither oc nor kubectl is found.
        """
        if self._cli_path is not None:
            return self._cli_path

        # Prefer oc over kubectl since we're on OpenShift
        for cli in ("oc", "kubectl"):
            path = shutil.which(cli)
            if path:
                self._cli_path = path
                logger.debug(f"Found CLI: {path}")
                return path

        raise PortForwardError(
            "Neither 'oc' nor 'kubectl' found in PATH. "
            "Please install the OpenShift CLI (oc) or Kubernetes CLI (kubectl)."
        )

    async def _wait_for_port_ready(
        self,
        port: int,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ) -> bool:
        """Wait for a local port to become available.

        Args:
            port: The local port to check.
            timeout: Maximum time to wait in seconds.
            poll_interval: Time between checks in seconds.

        Returns:
            True if port is ready, False if timeout.
        """
        elapsed = 0.0
        while elapsed < timeout:
            try:
                # Try to connect to the port
                _reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", port),
                    timeout=1.0,
                )
                writer.close()
                await writer.wait_closed()
                return True
            except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

        return False

    async def forward(
        self,
        namespace: str,
        service_name: str,
        remote_port: int,
        timeout: float = 10.0,
        use_https: bool = False,
    ) -> PortForwardConnection:
        """Start or reuse a port-forward connection.

        Args:
            namespace: Kubernetes namespace.
            service_name: Name of the service to forward to.
            remote_port: Port on the service to forward.
            timeout: Timeout in seconds waiting for connection.
            use_https: Whether the remote service uses HTTPS.

        Returns:
            PortForwardConnection with local URL information.

        Raises:
            PortForwardError: If port-forward fails to start.
        """
        key = (namespace, service_name, remote_port)
        lock = await self._get_lock()

        async with lock:
            # Check for existing connection
            if key in self._connections:
                conn = self._connections[key]
                # Check if process is still running
                if conn.process.returncode is None:
                    conn.ref_count += 1
                    logger.debug(
                        f"Reusing port-forward to {service_name}.{namespace}:{remote_port} "
                        f"(refcount={conn.ref_count})"
                    )
                    return conn
                else:
                    # Process died, remove stale connection
                    del self._connections[key]

            # Find CLI
            cli = self._find_oc_or_kubectl()

            # Allocate local port
            local_port = self._find_free_port()

            # Build command
            # Format: oc port-forward -n <namespace> svc/<service> <local>:<remote>
            cmd = [
                cli,
                "port-forward",
                "-n",
                namespace,
                f"svc/{service_name}",
                f"{local_port}:{remote_port}",
            ]

            logger.info(
                f"Starting port-forward: {service_name}.{namespace}:{remote_port} -> "
                f"localhost:{local_port}"
            )

            try:
                # Start subprocess
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # Wait for port to be ready
                if not await self._wait_for_port_ready(local_port, timeout):
                    # Check if process failed
                    if process.returncode is not None:
                        _, stderr = await process.communicate()
                        error_msg = stderr.decode() if stderr else "Unknown error"
                        raise PortForwardError(f"Port-forward process exited: {error_msg}")
                    else:
                        # Timeout but process still running - kill it
                        process.terminate()
                        await process.wait()
                        raise PortForwardError(
                            f"Timeout waiting for port-forward to "
                            f"{service_name}.{namespace}:{remote_port}"
                        )

                # Create connection object
                conn = PortForwardConnection(
                    namespace=namespace,
                    service_name=service_name,
                    remote_port=remote_port,
                    local_port=local_port,
                    process=process,
                    use_https=use_https,
                )

                self._connections[key] = conn
                logger.info(f"Port-forward established: {conn.local_url}")
                return conn

            except PortForwardError:
                raise
            except Exception as e:
                raise PortForwardError(f"Failed to start port-forward: {e}") from e

    async def release(self, conn: PortForwardConnection) -> None:
        """Release a port-forward connection.

        Decrements the reference count and terminates the subprocess
        if no more references exist.

        Args:
            conn: The connection to release.
        """
        key = (conn.namespace, conn.service_name, conn.remote_port)
        lock = await self._get_lock()

        async with lock:
            if key not in self._connections:
                return

            stored_conn = self._connections[key]
            stored_conn.ref_count -= 1

            if stored_conn.ref_count <= 0:
                logger.info(
                    f"Closing port-forward to "
                    f"{conn.service_name}.{conn.namespace}:{conn.remote_port}"
                )
                if stored_conn.process.returncode is None:
                    stored_conn.process.terminate()
                    try:
                        await asyncio.wait_for(stored_conn.process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        stored_conn.process.kill()
                        await stored_conn.process.wait()
                del self._connections[key]

    async def close_all(self) -> None:
        """Close all port-forward connections.

        Should be called during server shutdown.
        """
        lock = await self._get_lock()

        async with lock:
            for _key, conn in list(self._connections.items()):
                logger.info(
                    f"Closing port-forward to "
                    f"{conn.service_name}.{conn.namespace}:{conn.remote_port}"
                )
                if conn.process.returncode is None:
                    conn.process.terminate()
                    try:
                        await asyncio.wait_for(conn.process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        conn.process.kill()
                        await conn.process.wait()

            self._connections.clear()
            logger.info("All port-forward connections closed")

    @property
    def active_connections(self) -> int:
        """Get the number of active port-forward connections."""
        return len(self._connections)
