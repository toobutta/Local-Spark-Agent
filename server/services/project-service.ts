import { storage } from '../storage';
import type { Project } from '../../shared/schema';

export interface CreateProjectRequest {
  name: string;
  path?: string;
  settings?: Record<string, any>;
}

export class ProjectService {
  async getProjects(): Promise<Project[]> {
    return await storage.getProjects();
  }

  async getProject(id: string): Promise<Project | null> {
    return await storage.getProject(id);
  }

  async createProject(request: CreateProjectRequest): Promise<Project> {
    const project: any = {
      id: `project_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
      name: request.name,
      path: request.path || null,
      settings: request.settings || null,
      createdAt: new Date(),
      updatedAt: new Date()
    };

    return await storage.createProject(project);
  }

  async updateProject(id: string, updates: any): Promise<Project | null> {
    return await storage.updateProject(id, updates);
  }

  async deleteProject(id: string): Promise<boolean> {
    return await storage.deleteProject(id);
  }
}

// Singleton instance
export const projectService = new ProjectService();
