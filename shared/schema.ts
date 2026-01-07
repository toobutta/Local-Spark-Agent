import { sql } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, jsonb, integer, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

// Agents table
export const agents = pgTable("agents", {
  id: varchar("id").primaryKey(),
  name: text("name").notNull(),
  role: text("role").notNull(),
  status: text("status").default("idle"),
  config: jsonb("config"),
  createdAt: timestamp("created_at").defaultNow(),
  lastActive: timestamp("last_active"),
});

export const insertAgentSchema = createInsertSchema(agents).pick({
  id: true,
  name: true,
  role: true,
  status: true,
  config: true,
});

export type InsertAgent = z.infer<typeof insertAgentSchema>;
export type Agent = typeof agents.$inferSelect;

// Projects table
export const projects = pgTable("projects", {
  id: varchar("id").primaryKey(),
  name: text("name").notNull(),
  path: text("path"),
  settings: jsonb("settings"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertProjectSchema = createInsertSchema(projects).pick({
  id: true,
  name: true,
  path: true,
  settings: true,
});

export type InsertProject = z.infer<typeof insertProjectSchema>;
export type Project = typeof projects.$inferSelect;

// Command history table
export const commandHistory = pgTable("command_history", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  command: text("command").notNull(),
  output: text("output"),
  success: boolean("success").default(false),
  error: text("error"),
  executedAt: timestamp("executed_at").defaultNow(),
});

export const insertCommandHistorySchema = createInsertSchema(commandHistory).pick({
  id: true,
  command: true,
  output: true,
  success: true,
  error: true,
});

export type InsertCommandHistory = z.infer<typeof insertCommandHistorySchema>;
export type CommandHistory = typeof commandHistory.$inferSelect;

// DGX configurations table
export const dgxConfigs = pgTable("dgx_configs", {
  id: varchar("id").primaryKey(),
  name: text("name").notNull(),
  host: text("host").notNull(),
  port: integer("port").default(22),
  username: text("username"),
  sshKeyPath: text("ssh_key_path"),
  isDefault: boolean("is_default").default(false),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertDgxConfigSchema = createInsertSchema(dgxConfigs).pick({
  id: true,
  name: true,
  host: true,
  port: true,
  username: true,
  sshKeyPath: true,
  isDefault: true,
});

export type InsertDgxConfig = z.infer<typeof insertDgxConfigSchema>;
export type DgxConfig = typeof dgxConfigs.$inferSelect;
