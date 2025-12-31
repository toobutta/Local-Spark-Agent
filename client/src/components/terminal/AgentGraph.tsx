import { motion } from "framer-motion";
import { Bot, Zap, Shield, Database } from "lucide-react";

interface Agent {
  id: string;
  name: string;
  status: "idle" | "active" | "error";
  role: string;
}

interface AgentGraphProps {
  agents: Agent[];
}

export function AgentGraph({ agents }: AgentGraphProps) {
  if (agents.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-40 text-muted-foreground border border-dashed border-border rounded-md">
        <Bot className="w-8 h-8 mb-2 opacity-20" />
        <span className="text-xs">NO ACTIVE AGENTS</span>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {agents.map((agent, index) => (
        <motion.div
          key={agent.id}
          initial={{ x: 20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: index * 0.1 }}
          className="flex items-center gap-3 p-3 bg-card/50 border border-border rounded-md group hover:border-primary/50 transition-colors"
        >
          <div className={`
            p-2 rounded-full 
            ${agent.status === 'active' ? 'bg-primary/20 text-primary animate-pulse' : 
              agent.status === 'error' ? 'bg-destructive/20 text-destructive' : 'bg-muted text-muted-foreground'}
          `}>
            {agent.role === 'security' ? <Shield size={16} /> :
             agent.role === 'data' ? <Database size={16} /> :
             agent.role === 'compute' ? <Zap size={16} /> :
             <Bot size={16} />}
          </div>
          
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between">
              <span className="font-mono text-sm font-bold truncate group-hover:text-primary transition-colors">
                {agent.name}
              </span>
              <span className={`text-[10px] uppercase tracking-wider ${
                agent.status === 'active' ? 'text-primary' : 
                agent.status === 'error' ? 'text-destructive' : 'text-muted-foreground'
              }`}>
                {agent.status}
              </span>
            </div>
            <div className="h-1 w-full bg-background rounded-full mt-1 overflow-hidden">
              <motion.div 
                className={`h-full ${agent.status === 'active' ? 'bg-primary' : 'bg-muted-foreground'}`}
                initial={{ width: "0%" }}
                animate={{ width: agent.status === 'active' ? "100%" : "30%" }}
                transition={{ 
                  duration: agent.status === 'active' ? 2 : 0, 
                  repeat: agent.status === 'active' ? Infinity : 0 
                }}
              />
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
