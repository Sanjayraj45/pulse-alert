import asyncio
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store connected WebSocket clients
connected_clients = set()
latest_results = {}

async def broadcast(message: dict):
    """Send message to all connected WebSocket clients"""
    if connected_clients:
        data = json.dumps(message)
        dead = set()
        for client in connected_clients:
            try:
                await client.send_text(data)
            except Exception:
                dead.add(client)
        connected_clients.difference_update(dead)

def start_scheduler():
    """Placeholder - no auto scheduler needed, CSV driven"""
    logger.info("🏥 PulseAlert ready — CSV upload mode active")