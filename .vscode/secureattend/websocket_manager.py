from fastapi import WebSocket, WebSocketDisconnect
import json
from typing import List, Dict, Any
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            # Connection might be closed
            self.disconnect(websocket)

    async def broadcast(self, data: Dict[Any, Any]):
        """Broadcast message to all connected clients"""
        message = json.dumps(data)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_presence_update(self, student_id: int, status: str, student_name: str):
        """Broadcast presence update to all clients"""
        await self.broadcast({
            "type": "presence",
            "payload": {
                "student_id": student_id,
                "status": status,
                "student_name": student_name,
                "timestamp": str(asyncio.get_event_loop().time())
            }
        })

    async def broadcast_timeline_update(self, student_id: int, event_data: dict):
        """Broadcast timeline update to all clients"""
        await self.broadcast({
            "type": "timeline", 
            "payload": {
                "student_id": student_id,
                "event": event_data
            }
        })

    async def broadcast_leaderboard_update(self, metric: str):
        """Broadcast leaderboard update to all clients"""
        await self.broadcast({
            "type": "leaderboard",
            "payload": {
                "metric": metric,
                "updated_at": str(asyncio.get_event_loop().time())
            }
        })

    async def broadcast_insight_update(self, insight_data: dict):
        """Broadcast new insight to all clients"""
        await self.broadcast({
            "type": "insight",
            "payload": insight_data
        })

# Global manager instance
manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive by waiting for messages
            data = await websocket.receive_text()
            # Echo back for testing
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)