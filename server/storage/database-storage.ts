import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { eq } from 'drizzle-orm';
import {
  users,
  agents,
  projects,
  commandHistory,
  dgxConfigs,
} from '../../shared/schema';
import type {
  User,
  InsertUser,
  Agent,
  InsertAgent,
  Project,
  InsertProject,
  CommandHistory,
  InsertCommandHistory,
  DgxConfig,
  InsertDgxConfig,
} from '../../shared/schema';

// Define interface locally to avoid circular imports
interface IStorage {
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

export class DatabaseStorage implements IStorage {
  private db: ReturnType<typeof drizzle>;

  constructor() {
    if (!process.env.DATABASE_URL) {
      throw new Error('DATABASE_URL environment variable is required for database storage');
    }

    const client = postgres(process.env.DATABASE_URL);
    this.db = drizzle(client);
  }

  // User methods
  async getUser(id: string): Promise<User | undefined> {
    const result = await this.db.select().from(users).where(eq(users.id, id)).limit(1);
    return result[0];
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const result = await this.db.select().from(users).where(eq(users.username, username)).limit(1);
    return result[0];
  }

  async createUser(user: InsertUser): Promise<User> {
    const result = await this.db.insert(users).values(user).returning();
    if (!result[0]) {
      throw new Error('Failed to create user');
    }
    return result[0];
  }
  // Agent methods
  async getAgents(): Promise<Agent[]> {
    return await this.db.select().from(agents);
  }

  async getAgent(id: string): Promise<Agent | null> {
    const result = await this.db.select().from(agents).where(eq(agents.id, id)).limit(1);
    return result[0] || null;
  }

  async createAgent(agent: InsertAgent): Promise<Agent> {
    const result = await this.db.insert(agents).values(agent).returning();
    return result[0];
  }

  async updateAgent(id: string, updates: Partial<InsertAgent>): Promise<Agent | null> {
    const result = await this.db
      .update(agents)
      .set(updates)
      .where(eq(agents.id, id))
      .returning();
    return result[0] || null;
  }

  async deleteAgent(id: string): Promise<boolean> {
    const result = await this.db.delete(agents).where(eq(agents.id, id)).returning({ id: agents.id });
    return result.length > 0;
  }

  // Project methods
  async getProjects(): Promise<Project[]> {
    return await this.db.select().from(projects);
  }

  async getProject(id: string): Promise<Project | null> {
    const result = await this.db.select().from(projects).where(eq(projects.id, id)).limit(1);
    return result[0] || null;
  }

  async createProject(project: InsertProject): Promise<Project> {
    const result = await this.db.insert(projects).values(project).returning();
    return result[0];
  }

  async updateProject(id: string, updates: Partial<InsertProject>): Promise<Project | null> {
    const result = await this.db
      .update(projects)
      .set(updates)
      .where(eq(projects.id, id))
      .returning();
    return result[0] || null;
  }

  async deleteProject(id: string): Promise<boolean> {
    const result = await this.db.delete(projects).where(eq(projects.id, id)).returning({ id: projects.id });
    return result.length > 0;
  }

  // Command history methods
  async getCommandHistory(limit: number = 50): Promise<CommandHistory[]> {
    return await this.db
      .select()
      .from(commandHistory)
      .orderBy(commandHistory.executedAt)
      .limit(limit);
  }

  async createCommandHistory(entry: InsertCommandHistory): Promise<CommandHistory> {
    const result = await this.db.insert(commandHistory).values(entry).returning();
    return result[0];
  }

  // DGX config methods
  async getDgxConfigs(): Promise<DgxConfig[]> {
    return await this.db.select().from(dgxConfigs);
  }

  async getDgxConfig(id: string): Promise<DgxConfig | null> {
    const result = await this.db.select().from(dgxConfigs).where(eq(dgxConfigs.id, id)).limit(1);
    return result[0] || null;
  }

  async getDefaultDgxConfig(): Promise<DgxConfig | null> {
    const result = await this.db.select().from(dgxConfigs).where(eq(dgxConfigs.isDefault, true)).limit(1);
    return result[0] || null;
  }

  async createDgxConfig(config: InsertDgxConfig): Promise<DgxConfig> {
    const result = await this.db.insert(dgxConfigs).values(config).returning();
    return result[0];
  }

  async updateDgxConfig(id: string, updates: Partial<InsertDgxConfig>): Promise<DgxConfig | null> {
    const result = await this.db
      .update(dgxConfigs)
      .set(updates)
      .where(eq(dgxConfigs.id, id))
      .returning();
    return result[0] || null;
  }

  async deleteDgxConfig(id: string): Promise<boolean> {
    const result = await this.db.delete(dgxConfigs).where(eq(dgxConfigs.id, id)).returning({ id: dgxConfigs.id });
    return result.length > 0;
  }
}
