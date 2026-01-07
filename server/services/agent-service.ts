import { storage } from '../storage';
import { wsManager } from '../websocket';
import type { Agent } from '../../shared/schema';

export interface CreateAgentRequest {
  name: string;
  role: string;
  config?: Record<string, any>;
}

export class AgentService {
  async getAgents(): Promise<Agent[]> {
    return await storage.getAgents();
  }

  async getAgent(id: string): Promise<Agent | null> {
    return await storage.getAgent(id);
  }

  async createAgent(request: CreateAgentRequest): Promise<Agent> {
    const agent: any = {
      id: `agent_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
      name: request.name,
      role: request.role,
      status: 'active',
      config: request.config || null,
      createdAt: new Date(),
      lastActive: null
    };

    const createdAgent = await storage.createAgent(agent);

    // Broadcast agent creation
    wsManager.broadcast('agent_created', createdAgent);

    return createdAgent;
  }

  async updateAgent(id: string, updates: any): Promise<Agent | null> {
    const updatedAgent = await storage.updateAgent(id, updates);
    if (updatedAgent) {
      // Broadcast agent update
      wsManager.broadcast('agent_updated', updatedAgent);
    }
    return updatedAgent;
  }

  async deleteAgent(id: string): Promise<boolean> {
    const success = await storage.deleteAgent(id);
    if (success) {
      // Broadcast agent deletion
      wsManager.broadcast('agent_deleted', { id });
    }
    return success;
  }

  async stopAgent(id: string): Promise<boolean> {
    return this.updateAgent(id, { status: 'idle' }).then(agent => agent !== null);
  }
}

// Singleton instance
export const agentService = new AgentService();
