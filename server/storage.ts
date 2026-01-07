import { type User, type InsertUser, type Agent, type InsertAgent, type Project, type InsertProject, type CommandHistory, type InsertCommandHistory, type DgxConfig, type InsertDgxConfig } from "@shared/schema";
import { randomUUID } from "crypto";
import { DatabaseStorage } from "./storage/database-storage";

// modify the interface with any CRUD methods
// you might need

export interface IStorage {
  // User methods
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;

  // Agent methods
  getAgents(): Promise<Agent[]>;
  getAgent(id: string): Promise<Agent | null>;
  createAgent(agent: InsertAgent): Promise<Agent>;
  updateAgent(id: string, updates: Partial<InsertAgent>): Promise<Agent | null>;
  deleteAgent(id: string): Promise<boolean>;

  // Project methods
  getProjects(): Promise<Project[]>;
  getProject(id: string): Promise<Project | null>;
  createProject(project: InsertProject): Promise<Project>;
  updateProject(id: string, updates: Partial<InsertProject>): Promise<Project | null>;
  deleteProject(id: string): Promise<boolean>;

  // Command history methods
  getCommandHistory(limit?: number): Promise<CommandHistory[]>;
  createCommandHistory(entry: InsertCommandHistory): Promise<CommandHistory>;

  // DGX config methods
  getDgxConfigs(): Promise<DgxConfig[]>;
  getDgxConfig(id: string): Promise<DgxConfig | null>;
  getDefaultDgxConfig(): Promise<DgxConfig | null>;
  createDgxConfig(config: InsertDgxConfig): Promise<DgxConfig>;
  updateDgxConfig(id: string, updates: Partial<InsertDgxConfig>): Promise<DgxConfig | null>;
  deleteDgxConfig(id: string): Promise<boolean>;
}

export class MemStorage implements IStorage {
  private users: Map<string, User>;
  private agents: Map<string, Agent>;
  private projects: Map<string, Project>;
  private commandHistory: CommandHistory[];
  private dgxConfigs: Map<string, DgxConfig>;

  constructor() {
    this.users = new Map();
    this.agents = new Map();
    this.projects = new Map();
    this.commandHistory = [];
    this.dgxConfigs = new Map();
  }

  // User methods
  async getUser(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = randomUUID();
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  // Agent methods
  async getAgents(): Promise<Agent[]> {
    return Array.from(this.agents.values());
  }

  async getAgent(id: string): Promise<Agent | null> {
    return this.agents.get(id) || null;
  }

  async createAgent(agent: InsertAgent): Promise<Agent> {
    const agentWithDefaults: any = {
      ...agent,
      createdAt: new Date(),
      lastActive: null,
    };
    this.agents.set(agent.id, agentWithDefaults);
    return agentWithDefaults;
  }

  async updateAgent(id: string, updates: Partial<InsertAgent>): Promise<Agent | null> {
    const existing = this.agents.get(id);
    if (!existing) return null;

    const updated = { ...existing, ...updates, lastActive: new Date() };
    this.agents.set(id, updated);
    return updated;
  }

  async deleteAgent(id: string): Promise<boolean> {
    return this.agents.delete(id);
  }

  // Project methods
  async getProjects(): Promise<Project[]> {
    return Array.from(this.projects.values());
  }

  async getProject(id: string): Promise<Project | null> {
    return this.projects.get(id) || null;
  }

  async createProject(project: InsertProject): Promise<Project> {
    const projectWithDefaults: any = {
      ...project,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    this.projects.set(project.id, projectWithDefaults);
    return projectWithDefaults;
  }

  async updateProject(id: string, updates: Partial<InsertProject>): Promise<Project | null> {
    const existing = this.projects.get(id);
    if (!existing) return null;

    const updated = { ...existing, ...updates, updatedAt: new Date() };
    this.projects.set(id, updated);
    return updated;
  }

  async deleteProject(id: string): Promise<boolean> {
    return this.projects.delete(id);
  }

  // Command history methods
  async getCommandHistory(limit: number = 50): Promise<CommandHistory[]> {
    return this.commandHistory.slice(-limit);
  }

  async createCommandHistory(entry: InsertCommandHistory): Promise<CommandHistory> {
    const entryWithDefaults: any = {
      ...entry,
      executedAt: new Date(),
    };
    this.commandHistory.push(entryWithDefaults);
    return entryWithDefaults;
  }

  // DGX config methods
  async getDgxConfigs(): Promise<DgxConfig[]> {
    return Array.from(this.dgxConfigs.values());
  }

  async getDgxConfig(id: string): Promise<DgxConfig | null> {
    return this.dgxConfigs.get(id) || null;
  }

  async getDefaultDgxConfig(): Promise<DgxConfig | null> {
    return Array.from(this.dgxConfigs.values()).find(config => config.isDefault) || null;
  }

  async createDgxConfig(config: InsertDgxConfig): Promise<DgxConfig> {
    const configWithDefaults: any = {
      ...config,
      createdAt: new Date(),
    };
    this.dgxConfigs.set(config.id, configWithDefaults);
    return configWithDefaults;
  }

  async updateDgxConfig(id: string, updates: Partial<InsertDgxConfig>): Promise<DgxConfig | null> {
    const existing = this.dgxConfigs.get(id);
    if (!existing) return null;

    const updated = { ...existing, ...updates };
    this.dgxConfigs.set(id, updated);
    return updated;
  }

  async deleteDgxConfig(id: string): Promise<boolean> {
    return this.dgxConfigs.delete(id);
  }
}

// Use DatabaseStorage if DATABASE_URL is available, otherwise MemStorage
export const storage = process.env.DATABASE_URL
  ? new DatabaseStorage()
  : new MemStorage();
