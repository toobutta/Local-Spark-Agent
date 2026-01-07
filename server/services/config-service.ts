export interface SystemConfig {
  theme?: string;
  language?: string;
  notifications?: boolean;
  autoSave?: boolean;
  maxAgents?: number;
  defaultModel?: string;
  apiKeys?: Record<string, string>; // Will be encrypted in production
  dgx?: {
    host?: string;
    port?: number;
    username?: string;
  };
}

export class ConfigService {
  private config: SystemConfig = {
    theme: 'dark',
    language: 'en',
    notifications: true,
    autoSave: true,
    maxAgents: 10,
    defaultModel: 'gpt-4',
    dgx: {
      port: 22
    }
  };

  async getConfig(): Promise<SystemConfig> {
    // TODO: Load from database or file
    return this.config;
  }

  async updateConfig(updates: Partial<SystemConfig>): Promise<SystemConfig> {
    this.config = { ...this.config, ...updates };

    // TODO: Save to database or file

    return this.config;
  }

  async getConfigValue<T>(key: string): Promise<T | undefined> {
    const keys = key.split('.');
    let value: any = this.config;

    for (const k of keys) {
      value = value?.[k];
    }

    return value;
  }

  async setConfigValue(key: string, value: any): Promise<void> {
    const keys = key.split('.');
    let obj: any = this.config;

    for (let i = 0; i < keys.length - 1; i++) {
      const k = keys[i];
      if (!obj[k] || typeof obj[k] !== 'object') {
        obj[k] = {};
      }
      obj = obj[k];
    }

    obj[keys[keys.length - 1]] = value;

    // TODO: Save to database or file
  }

  async resetConfig(): Promise<SystemConfig> {
    this.config = {
      theme: 'dark',
      language: 'en',
      notifications: true,
      autoSave: true,
      maxAgents: 10,
      defaultModel: 'gpt-4',
      dgx: {
        port: 22
      }
    };

    return this.config;
  }
}

// Singleton instance
export const configService = new ConfigService();
