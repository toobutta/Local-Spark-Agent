import { useState, useRef, useEffect } from "react";
import { TerminalPrompt } from "@/components/terminal/TerminalPrompt";
import { LegoLoader } from "@/components/terminal/LegoLoader";
import { AICore } from "@/components/terminal/AICore";
import { AgentGraph } from "@/components/terminal/AgentGraph";
import { Terminal, Cpu, Network, Activity, Server, Command, Box, ShieldCheck, PlayCircle } from "lucide-react";
import { motion } from "framer-motion";

// Types
interface LogEntry {
  id: string;
  type: "command" | "output" | "system" | "error" | "success";
  content: string | React.ReactNode;
  timestamp: Date;
}

interface Agent {
  id: string;
  name: string;
  status: "idle" | "active" | "error";
  role: string;
}

export default function Home() {
  const [history, setHistory] = useState<LogEntry[]>([
    {
      id: "init",
      type: "system",
      content: "NEXUS CLI v2.4.0 INITIALIZED. CONNECTED TO LOCALHOST.",
      timestamp: new Date()
    }
  ]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeAgents, setActiveAgents] = useState<Agent[]>([]);
  const [dgxConnected, setDgxConnected] = useState(false);
  const outputRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [history]);

  const addLog = (type: LogEntry["type"], content: string | React.ReactNode) => {
    setHistory(prev => [...prev, {
      id: Math.random().toString(36).substring(7),
      type,
      content,
      timestamp: new Date()
    }]);
  };

  const handleCommand = async (cmd: string) => {
    const command = cmd.trim().toLowerCase();
    addLog("command", cmd);
    setIsProcessing(true);

    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 600));

    try {
      if (command === "help") {
        addLog("output", (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-primary font-bold mb-1">CORE COMMANDS</div>
              <ul className="space-y-1 text-muted-foreground">
                <li><span className="text-foreground">build &lt;project&gt;</span> - Compile project artifacts</li>
                <li><span className="text-foreground">deploy agent &lt;name&gt;</span> - Spawn autonomous agent</li>
                <li><span className="text-foreground">connect dgx</span> - Link to Nvidia DGX Spark</li>
                <li><span className="text-foreground">status</span> - System diagnostics</li>
                <li><span className="text-foreground">clear</span> - Clear terminal output</li>
              </ul>
            </div>
          </div>
        ));
      } else if (command.startsWith("build")) {
        addLog("system", "INITIATING BUILD SEQUENCE...");
        addLog("output", <LegoLoader />);
        await new Promise(resolve => setTimeout(resolve, 3000));
        addLog("success", "BUILD COMPLETE. ARTIFACTS DEPLOYED.");
      } else if (command.startsWith("deploy agent")) {
        const name = cmd.split(" ").slice(2).join(" ") || "Unnamed Agent";
        const roles = ["general", "security", "data", "compute"];
        const randomRole = roles[Math.floor(Math.random() * roles.length)];
        
        const newAgent: Agent = {
          id: Math.random().toString(),
          name: name.toUpperCase(),
          status: "active",
          role: randomRole
        };
        
        setActiveAgents(prev => [...prev, newAgent]);
        addLog("success", `AGENT [${newAgent.name}] DEPLOYED WITH ROLE [${newAgent.role.toUpperCase()}]`);
      } else if (command === "connect dgx") {
        addLog("system", "ESTABLISHING SECURE HANDSHAKE WITH NVIDIA DGX SPARK...");
        await new Promise(resolve => setTimeout(resolve, 1500));
        setDgxConnected(true);
        addLog("success", "CONNECTION ESTABLISHED. 8x A100 GPU CLUSTER AVAILABLE.");
      } else if (command === "status") {
        addLog("output", (
          <div className="flex flex-col gap-2 font-mono text-xs">
            <div className="flex justify-between"><span>CPU USAGE:</span> <span className="text-primary">12%</span></div>
            <div className="flex justify-between"><span>MEMORY:</span> <span className="text-primary">8.4GB / 32GB</span></div>
            <div className="flex justify-between"><span>DGX LINK:</span> <span className={dgxConnected ? "text-green-400" : "text-red-400"}>{dgxConnected ? "ONLINE" : "OFFLINE"}</span></div>
            <div className="flex justify-between"><span>ACTIVE AGENTS:</span> <span className="text-primary">{activeAgents.length}</span></div>
          </div>
        ));
      } else if (command === "clear") {
        setHistory([]);
      } else {
        addLog("error", `COMMAND NOT RECOGNIZED: ${command}`);
      }
    } catch (err) {
      addLog("error", "SYSTEM ERROR: COMMAND EXECUTION FAILED");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground font-mono overflow-hidden flex flex-col md:flex-row">
      {/* CRT Scanline Overlay */}
      <div className="scanline" />
      <div className="pointer-events-none fixed inset-0 z-50 bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.4)_100%)]" />

      {/* Main Terminal Area */}
      <div className="flex-1 flex flex-col h-screen relative z-10 border-r border-border/50">
        {/* Header */}
        <header className="h-12 border-b border-border/50 bg-card/20 flex items-center justify-between px-4 backdrop-blur-md">
          <div className="flex items-center gap-2 text-primary">
            <Terminal size={18} />
            <h1 className="font-display font-bold tracking-widest text-lg">NEXUS CLI</h1>
          </div>
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              LOCAL
            </div>
            <div className="font-mono">v2.4.0</div>
          </div>
        </header>

        {/* Output Area */}
        <div 
          ref={outputRef}
          className="flex-1 overflow-y-auto p-4 space-y-4 scroll-smooth"
        >
            {history.map((entry) => (
              <motion.div
                key={entry.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className={`text-sm ${
                  entry.type === 'command' ? 'text-muted-foreground font-bold pt-2' :
                  entry.type === 'error' ? 'text-destructive' :
                  entry.type === 'success' ? 'text-green-400' :
                  entry.type === 'system' ? 'text-blue-400 italic' :
                  'text-foreground'
                }`}
              >
                {entry.type === 'command' && <span className="mr-2 text-primary">{">"}</span>}
                {entry.content}
              </motion.div>
            ))}
          
          {isProcessing && (
             <div className="text-primary text-sm animate-pulse">_</div>
          )}
        </div>

        {/* Input Area */}
        <TerminalPrompt onCommand={handleCommand} disabled={isProcessing} />
      </div>

      {/* Sidebar / Status Panel */}
      <div className="w-full md:w-80 bg-card/10 border-l border-border/50 h-screen flex flex-col p-4 gap-6 overflow-y-auto relative z-10 backdrop-blur-sm">
        
        {/* AI Visual */}
        <div className="space-y-2">
          <h2 className="text-xs font-bold text-muted-foreground flex items-center gap-2">
            <Cpu size={14} /> SYSTEM CORE
          </h2>
          <AICore />
        </div>

        {/* System Stats */}
        <div className="space-y-3">
          <h2 className="text-xs font-bold text-muted-foreground flex items-center gap-2">
            <Activity size={14} /> DIAGNOSTICS
          </h2>
          <div className="space-y-2 text-xs font-mono">
            <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
              <div className="bg-primary h-full w-[45%]" />
            </div>
            <div className="flex justify-between text-muted-foreground">
              <span>CPU LOAD</span>
              <span>45%</span>
            </div>
            
            <div className="w-full bg-muted rounded-full h-2 overflow-hidden mt-2">
              <div className="bg-secondary h-full w-[72%]" />
            </div>
            <div className="flex justify-between text-muted-foreground">
              <span>MEMORY</span>
              <span>72%</span>
            </div>
          </div>
        </div>

        {/* Connections */}
        <div className="space-y-3">
          <h2 className="text-xs font-bold text-muted-foreground flex items-center gap-2">
            <Network size={14} /> CONNECTIONS
          </h2>
          <div className={`p-3 rounded border ${dgxConnected ? 'border-green-500/30 bg-green-500/5' : 'border-destructive/30 bg-destructive/5'} transition-all`}>
            <div className="flex items-center justify-between mb-2">
              <span className="font-bold text-sm">NVIDIA DGX SPARK</span>
              {dgxConnected ? <ShieldCheck size={14} className="text-green-500" /> : <div className="w-2 h-2 rounded-full bg-destructive" />}
            </div>
            <div className="text-[10px] text-muted-foreground">
              {dgxConnected ? "UPLINK SECURE • 400GB/s" : "DISCONNECTED"}
            </div>
          </div>
        </div>

        {/* Agent Orchestration */}
        <div className="space-y-2 flex-1">
          <h2 className="text-xs font-bold text-muted-foreground flex items-center gap-2">
            <Box size={14} /> ACTIVE AGENTS
          </h2>
          <AgentGraph agents={activeAgents} />
        </div>
      </div>
    </div>
  );
}
