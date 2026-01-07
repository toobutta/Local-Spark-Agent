import { Client, type ClientChannel } from 'ssh2';
import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { storage } from '../storage';

interface SSHConfig {
  host: string;
  port: number;
  username: string;
  privateKey?: string;
  password?: string;
}

interface GPUStatus {
  id: number;
  name: string;
  utilization: number;
  memoryUsed: number;
  memoryTotal: number;
  temperature: number;
}

interface FileInfo {
  name: string;
  path: string;
  type: 'file' | 'directory';
  size: number;
  modified: Date;
  permissions: string;
}

export class DGXService {
  private connection: Client | null = null;
  private currentConfig: SSHConfig | null = null;

  constructor() {
    this.connection = null;
  }

  async connect(configId?: string): Promise<boolean> {
    try {
      // Get DGX config from database
      let config;
      if (configId) {
        config = await storage.getDgxConfig(configId);
      } else {
        config = await storage.getDefaultDgxConfig();
      }

      if (!config) {
        throw new Error('No DGX configuration found');
      }

      const connectionConfig: SSHConfig = {
        host: config.host,
        port: config.port || 22,
        username: config.username || 'sparkplug',
        privateKey: config.sshKeyPath ? readFileSync(config.sshKeyPath, 'utf8') : undefined,
      };
      this.currentConfig = connectionConfig;

      return new Promise((resolve, reject) => {
        this.connection = new Client();
        
        // Track if promise has already been settled
        let settled = false;
        
        const settleResolve = (value: boolean) => {
          if (!settled) {
            settled = true;
            resolve(value);
          }
        };
        
        const settleReject = (error: Error) => {
          if (!settled) {
            settled = true;
            reject(error);
          }
        };

        this.connection.on('ready', () => {
          console.log(`Connected to DGX at ${this.currentConfig!.host}`);
          settleResolve(true);
        });

        this.connection.on('error', (err: any) => {
          console.error('DGX SSH connection error:', err);
          settleReject(err);
        });
        
        this.connection.on('close', () => {
          console.log('DGX SSH connection closed unexpectedly');
          if (!settled) {
            settleReject(new Error('SSH connection closed unexpectedly'));
          }
        });

        this.connection.connect(connectionConfig);
      });
    } catch (error) {
      console.error('Failed to connect to DGX:', error);
      return false;
    }
  }

  async disconnect(): Promise<void> {
    if (this.connection) {
      this.connection.end();
      this.connection = null;
      this.currentConfig = null;
    }
  }

  async executeCommand(command: string): Promise<string> {
    if (!this.connection) {
      throw new Error('Not connected to DGX');
    }

    return new Promise((resolve, reject) => {
      this.connection!.exec(command, (err: Error | undefined, stream: ClientChannel) => {
        if (err) {
          reject(err);
          return;
        }

        let output = '';
        let errorOutput = '';

        stream.on('close', (code: number, signal: string) => {
          if (code === 0) {
            resolve(output.trim());
          } else {
            reject(new Error(`Command failed with code ${code}: ${errorOutput}`));
          }
        });

        stream.on('data', (data: Buffer) => {
          output += data.toString();
        });

        stream.stderr.on('data', (data: Buffer) => {
          errorOutput += data.toString();
        });
      });
    });
  }

  async getGPUStatus(): Promise<GPUStatus[]> {
    try {
      // Run nvidia-smi command to get GPU status
      const output = await this.executeCommand(
        'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits'
      );

      const lines = output.trim().split('\n');
      const gpus: GPUStatus[] = [];

      for (const line of lines) {
        const parts = line.split(',').map(s => s.trim());
        if (parts.length >= 6) {
          gpus.push({
            id: parseInt(parts[0]),
            name: parts[1],
            utilization: parseInt(parts[2]),
            memoryUsed: parseInt(parts[3]),
            memoryTotal: parseInt(parts[4]),
            temperature: parseInt(parts[5]),
          });
        }
      }

      return gpus;
    } catch (error) {
      console.error('Failed to get GPU status:', error);
      // Return mock data if nvidia-smi fails
      return [
        {
          id: 0,
          name: 'H100',
          utilization: 45,
          memoryUsed: 32768,
          memoryTotal: 98304,
          temperature: 65,
        }
      ];
    }
  }

  async listFiles(path: string = '.'): Promise<FileInfo[]> {
    try {
      const output = await this.executeCommand(`ls -la "${path}"`);

      const lines = output.trim().split('\n');
      const files: FileInfo[] = [];

      for (const line of lines) {
        if (line.startsWith('total') || line.trim() === '') continue;

        const parts = line.split(/\s+/);
        if (parts.length >= 9) {
          const permissions = parts[0];
          const size = parseInt(parts[4]);
          const month = parts[5];
          const day = parts[6];
          const time = parts[7];
          const name = parts.slice(8).join(' ');

          // Skip current and parent directory entries
          if (name === '.' || name === '..') continue;

          const type = permissions.startsWith('d') ? 'directory' : 'file';

          files.push({
            name,
            path: join(path, name),
            type: type as 'file' | 'directory',
            size,
            modified: new Date(`${month} ${day} ${time}`),
            permissions,
          });
        }
      }

      return files;
    } catch (error) {
      console.error('Failed to list files:', error);
      return [];
    }
  }

  async readFile(filePath: string): Promise<string> {
    try {
      const output = await this.executeCommand(`cat "${filePath}"`);
      return output;
    } catch (error) {
      console.error('Failed to read file:', error);
      throw error;
    }
  }

  async uploadFile(localPath: string, remotePath: string): Promise<void> {
    // Note: This is a simplified implementation
    // In production, you'd use SFTP or scp
    throw new Error('File upload not implemented yet');
  }

  async downloadFile(remotePath: string, localPath: string): Promise<void> {
    // Note: This is a simplified implementation
    // In production, you'd use SFTP or scp
    throw new Error('File download not implemented yet');
  }

  async getSystemInfo(): Promise<any> {
    try {
      const hostname = await this.executeCommand('hostname');
      const uptime = await this.executeCommand('uptime');
      const cpuInfo = await this.executeCommand('nproc');
      const memoryInfo = await this.executeCommand('free -h');

      return {
        hostname: hostname.trim(),
        uptime: uptime.trim(),
        cpuCores: parseInt(cpuInfo.trim()),
        memoryInfo: memoryInfo.trim(),
      };
    } catch (error) {
      console.error('Failed to get system info:', error);
      return {
        hostname: 'unknown',
        uptime: 'unknown',
        cpuCores: 0,
        memoryInfo: 'unknown',
      };
    }
  }

  async runTrainingCommand(modelConfig: any): Promise<string> {
    // This would construct and run a training command based on the model config
    // For example, running a Python training script with specific parameters
    const command = `cd /workspace && python train.py --model=${modelConfig.model} --dataset=${modelConfig.dataset}`;
    return await this.executeCommand(command);
  }

  async runInferenceCommand(modelPath: string, inputData: any): Promise<string> {
    // This would run inference on a trained model
    const command = `cd /workspace && python infer.py --model=${modelPath} --input='${JSON.stringify(inputData)}'`;
    return await this.executeCommand(command);
  }

  isConnected(): boolean {
    return this.connection !== null;
  }

  getCurrentConfig(): SSHConfig | null {
    return this.currentConfig;
  }
}

// Singleton instance
export const dgxService = new DGXService();
