import type { Express } from "express";
import { type Server } from "http";
import { storage } from "./storage";
import { commandExecutor } from "./services/command-executor";
import { agentService } from "./services/agent-service";
import { projectService } from "./services/project-service";
import { configService } from "./services/config-service";
import { dgxService } from "./services/dgx-service";
import { mcpManager } from "./mcp";

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  // put application routes here
  // prefix all routes with /api

  // use storage to perform CRUD operations on the storage interface
  // e.g. storage.insertUser(user) or storage.getUserByUsername(username)

  // Command Execution API
  app.post('/api/commands/execute', async (req, res) => {
    try {
      const { command } = req.body;
      if (!command || typeof command !== 'string') {
        return res.status(400).json({ error: 'Command is required' });
      }

      const result = await commandExecutor.executeCommand(command);
      res.json(result);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to execute command', message: error.message });
    }
  });

  // MCP Status API
  app.get('/api/mcp/status', (_req, res) => {
    try {
      const status = mcpManager.getServers();
      res.json(status);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to get MCP status', message: error.message });
    }
  });

  // Agent Management API
  app.get('/api/agents', async (req, res) => {
    try {
      const agents = await agentService.getAgents();
      res.json(agents);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to get agents', message: error.message });
    }
  });

  app.get('/api/agents/:id', async (req, res) => {
    try {
      const { id } = req.params;
      const agent = await agentService.getAgent(id);
      if (!agent) {
        return res.status(404).json({ error: 'Agent not found' });
      }
      res.json(agent);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to get agent', message: error.message });
    }
  });

  app.post('/api/agents', async (req, res) => {
    try {
      const { name, role, config } = req.body;
      if (!name || !role) {
        return res.status(400).json({ error: 'Name and role are required' });
      }

      const agent = await agentService.createAgent({ name, role, config });
      res.status(201).json(agent);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to create agent', message: error.message });
    }
  });

  app.put('/api/agents/:id', async (req, res) => {
    try {
      const { id } = req.params;
      const updates = req.body;

      const agent = await agentService.updateAgent(id, updates);
      if (!agent) {
        return res.status(404).json({ error: 'Agent not found' });
      }
      res.json(agent);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to update agent', message: error.message });
    }
  });

  app.delete('/api/agents/:id', async (req, res) => {
    try {
      const { id } = req.params;
      const success = await agentService.deleteAgent(id);
      if (!success) {
        return res.status(404).json({ error: 'Agent not found' });
      }
      res.status(204).send();
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to delete agent', message: error.message });
    }
  });

  app.post('/api/agents/:id/stop', async (req, res) => {
    try {
      const { id } = req.params;
      const success = await agentService.stopAgent(id);
      if (!success) {
        return res.status(404).json({ error: 'Agent not found' });
      }
      res.json({ success: true });
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to stop agent', message: error.message });
    }
  });

  // Project Management API
  app.get('/api/projects', async (req, res) => {
    try {
      const projects = await projectService.getProjects();
      res.json(projects);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to get projects', message: error.message });
    }
  });

  app.get('/api/projects/:id', async (req, res) => {
    try {
      const { id } = req.params;
      const project = await projectService.getProject(id);
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }
      res.json(project);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to get project', message: error.message });
    }
  });

  app.post('/api/projects', async (req, res) => {
    try {
      const { name, path, settings } = req.body;
      if (!name) {
        return res.status(400).json({ error: 'Name is required' });
      }

      const project = await projectService.createProject({ name, path, settings });
      res.status(201).json(project);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to create project', message: error.message });
    }
  });

  app.put('/api/projects/:id', async (req, res) => {
    try {
      const { id } = req.params;
      const updates = req.body;

      const project = await projectService.updateProject(id, updates);
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }
      res.json(project);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to update project', message: error.message });
    }
  });

  app.delete('/api/projects/:id', async (req, res) => {
    try {
      const { id } = req.params;
      const success = await projectService.deleteProject(id);
      if (!success) {
        return res.status(404).json({ error: 'Project not found' });
      }
      res.status(204).send();
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to delete project', message: error.message });
    }
  });

  // Configuration API
  app.get('/api/config', async (req, res) => {
    try {
      const config = await configService.getConfig();
      res.json(config);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to get config', message: error.message });
    }
  });

  app.put('/api/config', async (req, res) => {
    try {
      const updates = req.body;
      const config = await configService.updateConfig(updates);
      res.json(config);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to update config', message: error.message });
    }
  });

  app.get('/api/config/:key', async (req, res) => {
    try {
      const { key } = req.params;
      const value = await configService.getConfigValue(key);
      if (value === undefined) {
        return res.status(404).json({ error: 'Config key not found' });
      }
      res.json({ key, value });
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to get config value', message: error.message });
    }
  });

  app.put('/api/config/:key', async (req, res) => {
    try {
      const { key } = req.params;
      const { value } = req.body;
      await configService.setConfigValue(key, value);
      res.json({ success: true });
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to set config value', message: error.message });
    }
  });

  app.post('/api/config/reset', async (req, res) => {
    try {
      const config = await configService.resetConfig();
      res.json(config);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to reset config', message: error.message });
    }
  });

  // DGX Service API
  app.get('/api/dgx/status', async (req, res) => {
    try {
      if (!dgxService.isConnected()) {
        return res.status(503).json({ error: 'Not connected to DGX' });
      }

      const gpuStatus = await dgxService.getGPUStatus();
      const systemInfo = await dgxService.getSystemInfo();

      res.json({
        connected: true,
        gpus: gpuStatus,
        system: systemInfo,
      });
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to get DGX status', message: error.message });
    }
  });

  app.post('/api/dgx/connect', async (req, res) => {
    try {
      const { configId } = req.body;
      const connected = await dgxService.connect(configId);

      if (connected) {
        res.json({ success: true, message: 'Connected to DGX' });
      } else {
        res.status(500).json({ error: 'Failed to connect to DGX' });
      }
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to connect to DGX', message: error.message });
    }
  });

  app.post('/api/dgx/disconnect', async (req, res) => {
    try {
      await dgxService.disconnect();
      res.json({ success: true, message: 'Disconnected from DGX' });
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to disconnect from DGX', message: error.message });
    }
  });

  app.post('/api/dgx/execute', async (req, res) => {
    try {
      const { command } = req.body;

      if (!command || typeof command !== 'string') {
        return res.status(400).json({ error: 'Command is required' });
      }

      if (!dgxService.isConnected()) {
        return res.status(503).json({ error: 'Not connected to DGX' });
      }

      const output = await dgxService.executeCommand(command);
      res.json({ output });
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to execute command on DGX', message: error.message });
    }
  });

  app.get('/api/dgx/files', async (req, res) => {
    try {
      const { path = '.' } = req.query;

      if (!dgxService.isConnected()) {
        return res.status(503).json({ error: 'Not connected to DGX' });
      }

      const files = await dgxService.listFiles(path as string);
      res.json({ files });
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to list DGX files', message: error.message });
    }
  });

  app.post('/api/dgx/train', async (req, res) => {
    try {
      const { modelConfig } = req.body;

      if (!modelConfig) {
        return res.status(400).json({ error: 'Model config is required' });
      }

      if (!dgxService.isConnected()) {
        return res.status(503).json({ error: 'Not connected to DGX' });
      }

      const output = await dgxService.runTrainingCommand(modelConfig);
      res.json({ output });
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to start training on DGX', message: error.message });
    }
  });

  app.post('/api/dgx/infer', async (req, res) => {
    try {
      const { modelPath, inputData } = req.body;

      if (!modelPath || !inputData) {
        return res.status(400).json({ error: 'Model path and input data are required' });
      }

      if (!dgxService.isConnected()) {
        return res.status(503).json({ error: 'Not connected to DGX' });
      }

      const output = await dgxService.runInferenceCommand(modelPath, inputData);
      res.json({ output });
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to run inference on DGX', message: error.message });
    }
  });

  // DGX Configuration API
  app.get('/api/dgx/configs', async (req, res) => {
    try {
      const configs = await storage.getDgxConfigs();
      res.json(configs);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to get DGX configs', message: error.message });
    }
  });

  app.post('/api/dgx/configs', async (req, res) => {
    try {
      const { name, host, port, username, sshKeyPath, isDefault } = req.body;

      if (!name || !host) {
        return res.status(400).json({ error: 'Name and host are required' });
      }

      const config = await storage.createDgxConfig({
        id: `dgx_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
        name,
        host,
        port: port || 22,
        username,
        sshKeyPath,
        isDefault: isDefault || false,
      });

      res.status(201).json(config);
    } catch (error: any) {
      res.status(500).json({ error: 'Failed to create DGX config', message: error.message });
    }
  });

  return httpServer;
}
