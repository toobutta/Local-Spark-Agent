import { WebSocketServer, WebSocket } from 'ws';
import type { Server } from 'http';

interface WebSocketClient {
  ws: WebSocket;
  id: string;
}

interface BroadcastMessage {
  type: string;
  data: any;
  timestamp?: Date;
}

export class WebSocketManager {
  private wss: WebSocketServer | null = null;
  private clients: Map<string, WebSocketClient> = new Map();
  private messageHandlers: Map<string, (data: any, clientId: string) => void> = new Map();

  constructor() {}

  setupWebSocket(server: Server): WebSocketServer {
    this.wss = new WebSocketServer({ server, path: '/ws' });

    this.wss.on('connection', (ws: WebSocket) => {
      const clientId = this.generateClientId();
      const client: WebSocketClient = { ws, id: clientId };

      this.clients.set(clientId, client);
      console.log(`WebSocket client connected: ${clientId}`);

      ws.on('message', (data: Buffer) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleMessage(message, clientId);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      });

      ws.on('close', () => {
        this.clients.delete(clientId);
        console.log(`WebSocket client disconnected: ${clientId}`);
      });

      ws.on('error', (error) => {
        console.error(`WebSocket error for client ${clientId}:`, error);
        this.clients.delete(clientId);
      });

      // Send welcome message
      ws.send(JSON.stringify({
        type: 'connected',
        data: { clientId },
        timestamp: new Date()
      }));
    });

    return this.wss;
  }

  private generateClientId(): string {
    return `client_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private handleMessage(message: any, clientId: string) {
    const { type, data } = message;

    if (this.messageHandlers.has(type)) {
      const handler = this.messageHandlers.get(type)!;
      handler(data, clientId);
    } else {
      console.log(`Unhandled WebSocket message type: ${type} from client ${clientId}`);
    }
  }

  // Register message handlers
  onMessage(type: string, handler: (data: any, clientId: string) => void) {
    this.messageHandlers.set(type, handler);
  }

  // Broadcast to all connected clients
  broadcast(type: string, data: any) {
    if (!this.wss) {
      console.warn('WebSocket server not initialized');
      return;
    }

    const message: BroadcastMessage = {
      type,
      data,
      timestamp: new Date()
    };

    const messageStr = JSON.stringify(message);

    this.wss.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(messageStr);
      }
    });
  }

  // Send to specific client
  sendToClient(clientId: string, type: string, data: any) {
    const client = this.clients.get(clientId);
    if (!client || client.ws.readyState !== WebSocket.OPEN) {
      return false;
    }

    const message: BroadcastMessage = {
      type,
      data,
      timestamp: new Date()
    };

    client.ws.send(JSON.stringify(message));
    return true;
  }

  // Get connected client count
  getClientCount(): number {
    return this.clients.size;
  }

  // Cleanup
  close() {
    if (this.wss) {
      this.wss.close();
      this.wss = null;
    }
    this.clients.clear();
    this.messageHandlers.clear();
  }
}

// Singleton instance
export const wsManager = new WebSocketManager();
