import { useState, useRef, useEffect } from "react";
import { Link } from "wouter";
import { TerminalPrompt } from "@/components/terminal/TerminalPrompt";
import { LegoLoader } from "@/components/terminal/LegoLoader";
import { AICore } from "@/components/terminal/AICore";
import { AgentGraph } from "@/components/terminal/AgentGraph";
import { MatrixLoader } from "@/components/terminal/MatrixLoader";
import { ActiveAgentsFeed } from "@/components/terminal/ActiveAgentsFeed";
import { SphereSpinner } from "@/components/terminal/SphereSpinner";
import { Terminal, Cpu, Network, Activity, Server, Command, Box, ShieldCheck, PlayCircle, Settings, FolderOpen, Brain } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

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

interface ThoughtLog {
  id: string;
  agent: string;
  thought: string;
  file?: string;
  timestamp: Date;
}

export default function Home() {
  const [loading, setLoading] = useState(true);
  const [history, setHistory] = useState<LogEntry[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeAgents, setActiveAgents] = useState<Agent[]>([]);
  const [dgxConnected, setDgxConnected] = useState(false);
  const [thoughtLogs, setThoughtLogs] = useState<ThoughtLog[]>([]);
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

  const addThought = (agent: string, thought: string, file?: string) => {
    setThoughtLogs(prev => [...prev, {
      id: Math.random().toString(36).substring(7),
      agent,
      thought,
      file,
      timestamp: new Date()
    }]);
  };

  const handleCommand = async (cmd: string) => {
    const command = cmd.trim().toLowerCase();
    addLog("command", cmd);
    setIsProcessing(true);

    // Simulate processing delay with spinning sphere
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
                <li><span className="text-foreground">research &lt;topic&gt;</span> - Initiate neural search</li>
                <li><span className="text-foreground">status</span> - System diagnostics</li>
                <li><span className="text-foreground">clear</span> - Clear terminal output</li>
              </ul>
            </div>
          </div>
        ));
      } else if (command.startsWith("research")) {
        const topic = command.split(" ").slice(1).join(" ") || "General Knowledge";
        addLog("system", `INITIATING NEURAL SEARCH: ${topic.toUpperCase()}...`);
        
        // Simulate Agent Thoughts
        const thoughts = [
          { msg: "Scanning local vector database...", file: "knowledge_base.vdb" },
          { msg: "Querying semantic index...", file: "index_shard_01.dat" },
          { msg: "Synthesizing research nodes...", file: "graph_builder.py" },
          { msg: "Optimizing results..." }
        ];

        for (const t of thoughts) {
          await new Promise(r => setTimeout(r, 800));
          addThought("RESEARCH-AGENT", t.msg, t.file);
        }

        addLog("success", "RESEARCH COMPLETE. DATA ASSIMILATED.");
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
        addThought(newAgent.name, "System interface established.", "init.sh");
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

  if (loading) {
    return <MatrixLoader onComplete={() => setLoading(false)} />;
  }

  return (
    <div className="min-h-screen bg-background text-foreground font-mono overflow-hidden flex flex-col md:flex-row">
      {/* CRT Scanline Overlay */}
      <div className="scanline" />
      <div className="pointer-events-none fixed inset-0 z-50 bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.4)_100%)]" />

      {/* Main Terminal Area */}
      <div className="flex-1 flex flex-col h-screen relative z-10 border-r border-border/50">
        {/* Header */}
        <header className="h-14 border-b border-border/50 bg-card/20 flex items-center justify-between px-4 backdrop-blur-md shrink-0">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-primary">
              <Terminal size={18} />
              <h1 className="font-display font-bold tracking-widest text-lg">NEXUS CLI</h1>
            </div>
            
            <div className="h-8 w-px bg-border/30 mx-2 hidden md:block" />
            
            <div className="hidden md:flex flex-col justify-center">
              <span className="text-[10px] text-primary/70 uppercase tracking-widest font-bold mb-0.5">Project: Genesis</span>
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground font-mono bg-black/20 px-2 py-0.5 rounded border border-white/5">
                <FolderOpen size={10} className="text-secondary" />
                <span>~/workspace/nexus-core/src</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-black/40 border border-white/5">
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse shadow-[0_0_5px_#22c55e]" />
                <span className="font-bold tracking-wider text-[10px]">LOCAL</span>
              </div>
              <span className="text-border/50">|</span>
              <span className="font-mono text-[10px] opacity-70">v2.4.0</span>
            </div>
            
            <Link href="/admin">
              <button 
                className="p-2 hover:bg-primary/10 hover:text-primary rounded-md transition-all duration-300 border border-white/5 hover:border-primary/30 hover:shadow-[0_0_10px_rgba(0,255,255,0.2)] group"
                title="Configurations"
              >
                <Settings size={18} className="group-hover:rotate-90 transition-transform duration-500" />
              </button>
            </Link>
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
          <div className="relative">
            <SphereSpinner isActive={isProcessing} />
          </div>
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
        <div className="space-y-2 flex-1 flex flex-col min-h-0">
          <h2 className="text-xs font-bold text-muted-foreground flex items-center gap-2">
            <Brain size={14} /> NEURAL LINK
          </h2>
          <div className="flex-1 min-h-0">
            <ActiveAgentsFeed logs={thoughtLogs} />
          </div>
        </div>
      </div>
    </div>
  );
}
