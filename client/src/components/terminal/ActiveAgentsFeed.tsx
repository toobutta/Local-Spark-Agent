import { motion, AnimatePresence } from "framer-motion";
import { FileSearch, Activity, Brain } from "lucide-react";

interface ThoughtLog {
  id: string;
  agent: string;
  thought: string;
  file?: string;
  timestamp: Date;
}

interface ActiveAgentsFeedProps {
  logs: ThoughtLog[];
}

export function ActiveAgentsFeed({ logs }: ActiveAgentsFeedProps) {
  return (
    <div className="h-full flex flex-col bg-black/40 rounded border border-green-500/20 overflow-hidden font-mono text-xs">
      <div className="px-3 py-2 bg-green-500/10 border-b border-green-500/20 flex items-center justify-between">
        <div className="flex items-center gap-2 text-green-400">
          <Brain size={14} />
          <span className="font-bold tracking-wider">NEURAL STREAM</span>
        </div>
        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
      </div>
      
      <div className="flex-1 overflow-hidden relative p-2 space-y-2">
        <AnimatePresence mode="popLayout">
          {logs.slice(-5).map((log) => (
            <motion.div
              key={log.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, height: 0 }}
              className="p-2 border-l-2 border-green-500/40 bg-green-500/5 hover:bg-green-500/10 transition-colors"
            >
              <div className="flex justify-between items-center mb-1 text-green-300/70 text-[10px]">
                <span>{log.agent}</span>
                <span>{log.timestamp.toLocaleTimeString()}</span>
              </div>
              
              <div className="text-green-100 mb-1 leading-tight">
                {log.thought}
              </div>
              
              {log.file && (
                <div className="flex items-center gap-1.5 text-green-400/80 bg-green-500/10 px-1.5 py-0.5 rounded w-fit mt-1">
                  <FileSearch size={10} />
                  <span className="truncate max-w-[150px]">{log.file}</span>
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
        
        {logs.length === 0 && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-green-500/30 gap-2">
            <Activity className="animate-pulse" size={24} />
            <span>AWAITING INPUT...</span>
          </div>
        )}
      </div>
    </div>
  );
}
