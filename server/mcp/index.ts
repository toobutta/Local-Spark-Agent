import { wsManager } from "../websocket";

export type MCPStatus = "online" | "offline" | "maintenance";

export interface MCPServerStatus {
  id: string;
  name: string;
  type: string;
  status: MCPStatus;
  latency: number | null;
  description?: string;
  lastChecked: Date | null;
}

class MCPManager {
  private servers: Map<string, MCPServerStatus>;
  private monitorInterval: NodeJS.Timeout | null = null;

  constructor() {
    this.servers = new Map();
    this.bootstrapDefaults();
  }

  private bootstrapDefaults() {
    const defaults: MCPServerStatus[] = [
      {
        id: "postgres_connector",
        name: "PostgreSQL Connector",
        type: "database",
        status: "online",
        latency: 12,
        description: "Direct SQL access for project datasets",
        lastChecked: null,
      },
      {
        id: "filesystem_watcher",
        name: "Filesystem Watcher",
        type: "system",
        status: "online",
        latency: 2,
        description: "Local project tree synchronization",
        lastChecked: null,
      },
      {
        id: "github_repo",
        name: "GitHub Repository",
        type: "vcs",
        status: "offline",
        latency: null,
        description: "GitHub MCP bridge",
        lastChecked: null,
      },
    ];

    defaults.forEach((server) => this.servers.set(server.id, server));
  }

  getServers(): MCPServerStatus[] {
    return Array.from(this.servers.values());
  }

  getServer(id: string): MCPServerStatus | undefined {
    return this.servers.get(id);
  }

  updateServerStatus(id: string, updates: Partial<Omit<MCPServerStatus, "id">>) {
    const existing = this.servers.get(id);
    if (!existing) return;

    const updated: MCPServerStatus = {
      ...existing,
      ...updates,
      lastChecked: updates.lastChecked ?? new Date(),
    };

    this.servers.set(id, updated);
    this.broadcastStatuses();
  }

  async startStatusMonitoring(intervalMs = 15000) {
    if (this.monitorInterval) return;

    await this.refreshStatuses();
    this.monitorInterval = setInterval(() => {
      void this.refreshStatuses();
    }, intervalMs);
  }

  stopStatusMonitoring() {
    if (this.monitorInterval) {
      clearInterval(this.monitorInterval);
      this.monitorInterval = null;
    }
  }

  private async refreshStatuses() {
    const now = new Date();

    this.servers.forEach((server, id) => {
      const nextStatus = this.randomizeStatus(server.status);
      const latency =
        nextStatus === "offline"
          ? null
          : Math.max(5, Math.round((server.latency ?? 15) + (Math.random() * 10 - 5)));

      this.servers.set(id, {
        ...server,
        status: nextStatus,
        latency,
        lastChecked: now,
      });
    });

    this.broadcastStatuses();
  }

  private randomizeStatus(current: MCPStatus): MCPStatus {
    const roll = Math.random();

    if (roll > 0.95) {
      return "maintenance";
    }

    if (roll > 0.9) {
      return "offline";
    }

    if (current === "offline" && roll > 0.5) {
      return "online";
    }

    if (current === "maintenance" && roll > 0.6) {
      return "online";
    }

    return current;
  }

  private broadcastStatuses() {
    wsManager.broadcast("mcp_status", {
      servers: this.getServers(),
    });
  }
}

export const mcpManager = new MCPManager();

