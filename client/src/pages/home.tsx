import { useState, useRef, useEffect } from "react";
import { Link, useLocation } from "wouter";
import { TerminalPrompt } from "@/components/terminal/TerminalPrompt";
import { LegoLoader } from "@/components/terminal/LegoLoader";
import { AICore } from "@/components/terminal/AICore";
import { AgentGraph } from "@/components/terminal/AgentGraph";
import { MatrixLoader } from "@/components/terminal/MatrixLoader";
import { ActiveAgentsFeed } from "@/components/terminal/ActiveAgentsFeed";
import { SphereSpinner } from "@/components/terminal/SphereSpinner";
import { Terminal, Cpu, Network, Activity, Server, Command, Box, ShieldCheck, PlayCircle, Settings, FolderOpen, Brain, Zap, HardDrive, FileCode, Code, Database, Braces, FileText, Download, Upload, RefreshCw, ChevronRight, ListTodo, Plus, Check, Clock } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Checkbox } from "@/components/ui/checkbox";
import { motion, AnimatePresence } from "framer-motion";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";

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
  const [history, setHistory] = useState<LogEntry[]>([
    {
      id: "init",
      type: "system",
      content: (
        <div className="font-mono text-xs leading-tight">
          <span className="text-green-500">
            ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗<br/>
            ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝<br/>
            ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗<br/>
            ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║<br/>
            ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║<br/>
            ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝<br/>
          </span>
          <br/>
          <span className="text-blue-400">System v2.4.0</span> <span className="text-muted-foreground">|</span> <span className="text-yellow-500">READY</span>
        </div>
      ),
      timestamp: new Date()
    }
  ]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeAgents, setActiveAgents] = useState<Agent[]>([]);
  const [dgxConnected, setDgxConnected] = useState(false);
  const [thoughtLogs, setThoughtLogs] = useState<ThoughtLog[]>([]);
  const [selectedCoreView, setSelectedCoreView] = useState("core");
  const outputRef = useRef<HTMLDivElement>(null);
  const [_, setLocation] = useLocation();

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
          <div className="font-mono text-sm space-y-2">
            <div className="text-primary font-bold border-b border-primary/30 pb-1 mb-2">AVAILABLE COMMANDS</div>
            <div className="grid grid-cols-[120px_1fr] gap-x-4 gap-y-1">
              <span className="text-yellow-500">build</span>      <span className="text-muted-foreground">Compile project artifacts</span>
              <span className="text-yellow-500">deploy</span>     <span className="text-muted-foreground">Spawn autonomous agent</span>
              <span className="text-yellow-500">connect</span>    <span className="text-muted-foreground">Link to external compute</span>
              <span className="text-yellow-500">research</span>   <span className="text-muted-foreground">Initiate neural search</span>
              <span className="text-yellow-500">status</span>     <span className="text-muted-foreground">System diagnostics</span>
              <span className="text-yellow-500">agents</span>     <span className="text-muted-foreground">List active agents</span>
              <span className="text-yellow-500">mcp</span>        <span className="text-muted-foreground">Show MCP connections</span>
              <span className="text-yellow-500">marketplace</span> <span className="text-muted-foreground">Browse plugins</span>
              <span className="text-yellow-500">plugins</span>    <span className="text-muted-foreground">Manage extensions</span>
              <span className="text-yellow-500">settings</span>   <span className="text-muted-foreground">Open config panel</span>
              <span className="text-yellow-500">clear</span>      <span className="text-muted-foreground">Clear terminal output</span>
            </div>
            <div className="mt-2 text-xs text-muted-foreground bg-white/5 p-2 rounded">
              <span className="text-blue-400">TIP:</span> You can also use <span className="text-white">/</span> prefix for commands (e.g. <span className="text-white">/settings</span>)
            </div>
          </div>
        ));
      } else if (command === "settings" || command === "/settings") {
        addLog("system", "REDIRECTING TO SYSTEM CONFIGURATION...");
        await new Promise(resolve => setTimeout(resolve, 800));
        setLocation("/admin");
      } else if (command === "agents" || command === "/agents") {
        addLog("output", (
          <div className="space-y-2 font-mono text-xs">
             <div className="text-blue-400 font-bold border-b border-blue-500/30 pb-1">ACTIVE AGENT SWARM</div>
             {activeAgents.length === 0 ? (
               <div className="text-muted-foreground italic">No agents deployed. Use 'deploy agent' to start one.</div>
             ) : (
               activeAgents.map(agent => (
                 <div key={agent.id} className="flex items-center gap-3 bg-white/5 p-2 rounded border border-white/5">
                   <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                   <span className="font-bold text-white">{agent.name}</span>
                   <span className="text-muted-foreground">[{agent.role}]</span>
                   <span className="ml-auto text-green-400">{agent.status}</span>
                 </div>
               ))
             )}
             <div className="text-[10px] text-muted-foreground mt-2">Total active nodes: {activeAgents.length}</div>
          </div>
        ));
      } else if (command === "mcp" || command === "/mcp") {
        addLog("output", (
          <div className="space-y-2 font-mono text-xs">
             <div className="text-purple-400 font-bold border-b border-purple-500/30 pb-1">MODEL CONTEXT PROTOCOL</div>
             <div className="grid grid-cols-2 gap-2">
                {[
                  { id: "github", status: "offline", latency: "-" },
                  { id: "postgres", status: "online", latency: "12ms" },
                  { id: "filesystem", status: "online", latency: "1ms" },
                  { id: "brave-search", status: "online", latency: "145ms" },
                ].map(mcp => (
                  <div key={mcp.id} className="bg-white/5 p-2 rounded border border-white/5 flex justify-between items-center">
                    <span className="text-white">{mcp.id}</span>
                    <span className={mcp.status === 'online' ? "text-green-400" : "text-red-400"}>{mcp.status.toUpperCase()}</span>
                  </div>
                ))}
             </div>
          </div>
        ));
      } else if (command === "marketplace" || command === "/marketplace" || command === "plugins" || command === "/plugins") {
        addLog("output", (
          <div className="space-y-2 font-mono text-xs">
             <div className="text-orange-400 font-bold border-b border-orange-500/30 pb-1">PLUGIN MARKETPLACE</div>
             <div className="space-y-1">
                <div className="flex justify-between items-center p-2 hover:bg-white/5 cursor-pointer rounded">
                   <span className="text-white font-bold">Research Assistant Pro</span>
                   <span className="text-green-400">INSTALLED</span>
                </div>
                <div className="flex justify-between items-center p-2 hover:bg-white/5 cursor-pointer rounded">
                   <span className="text-white font-bold">Python Code Interpreter</span>
                   <span className="text-green-400">INSTALLED</span>
                </div>
                <div className="flex justify-between items-center p-2 hover:bg-white/5 cursor-pointer rounded">
                   <span className="text-muted-foreground">Stripe Payments</span>
                   <span className="text-blue-400 hover:underline">INSTALL</span>
                </div>
                <div className="flex justify-between items-center p-2 hover:bg-white/5 cursor-pointer rounded">
                   <span className="text-muted-foreground">Slack Integration</span>
                   <span className="text-blue-400 hover:underline">INSTALL</span>
                </div>
             </div>
             <div className="text-[10px] text-muted-foreground mt-2 border-t border-white/5 pt-1">
               Use <span className="text-white">install &lt;plugin&gt;</span> to add new capabilities.
             </div>
          </div>
        ));
      } else if (command.startsWith("research")) {
        const topic = command.split(" ").slice(1).join(" ") || "General Knowledge";
        addLog("system", (
          <span className="text-cyan-400">
            [SYS] INITIATING NEURAL SEARCH<br/>
            <span className="text-muted-foreground">TARGET:</span> {topic.toUpperCase()}
          </span>
        ));
        
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

        addLog("success", (
          <span className="text-green-400">
             ✔ RESEARCH COMPLETE<br/>
             <span className="text-muted-foreground text-xs">Data assimilated into knowledge graph.</span>
          </span>
        ));
      } else if (command.startsWith("build")) {
        addLog("system", <span className="text-blue-400">➔ INITIATING BUILD SEQUENCE...</span>);
        addLog("output", <LegoLoader />);
        await new Promise(resolve => setTimeout(resolve, 3000));
        addLog("success", <span className="text-green-400">✔ BUILD COMPLETE. ARTIFACTS DEPLOYED.</span>);
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
        addLog("success", (
          <div className="bg-green-500/10 border border-green-500/30 p-2 rounded text-green-400">
            <div className="font-bold">AGENT DEPLOYED</div>
            <div className="text-xs grid grid-cols-2 mt-1 gap-4">
              <span>ID: {newAgent.id.substring(0,8)}</span>
              <span>ROLE: {newAgent.role.toUpperCase()}</span>
            </div>
          </div>
        ));
        addThought(newAgent.name, "System interface established.", "init.sh");
      } else if (command === "connect dgx") {
        addLog("system", "ESTABLISHING SECURE HANDSHAKE WITH NVIDIA DGX SPARK...");
        await new Promise(resolve => setTimeout(resolve, 1500));
        setDgxConnected(true);
        addLog("success", "CONNECTION ESTABLISHED. 8x A100 GPU CLUSTER AVAILABLE.");
      } else if (command === "status") {
        addLog("output", (
          <div className="flex flex-col gap-1 font-mono text-xs border border-border p-3 rounded bg-black/20">
            <div className="flex justify-between border-b border-white/5 pb-1 mb-1">
              <span className="font-bold text-muted-foreground">SYSTEM DIAGNOSTICS</span>
              <span className="text-green-500">NORMAL</span>
            </div>
            <div className="flex justify-between"><span>CPU USAGE</span> <span className="text-primary">▓▓▓░░░░░░░ 32%</span></div>
            <div className="flex justify-between"><span>MEMORY</span> <span className="text-primary">▓▓▓▓▓░░░░░ 8.4GB</span></div>
            <div className="flex justify-between"><span>DGX LINK</span> <span className={dgxConnected ? "text-green-400" : "text-red-400"}>{dgxConnected ? "● ONLINE" : "○ OFFLINE"}</span></div>
            <div className="flex justify-between"><span>AGENTS</span> <span className="text-blue-400">{activeAgents.length} ACTIVE</span></div>
          </div>
        ));
      } else if (command === "clear") {
        setHistory([]);
      } else {
        addLog("error", <span className="text-red-500">✖ COMMAND NOT RECOGNIZED: {command}</span>);
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

        {/* Footer Stats & Tools */}
        <div className="h-10 border-t border-white/10 bg-black/40 flex items-center justify-between px-4 text-[10px]">
          <div className="flex items-center gap-4 flex-1">
             <div className="flex items-center gap-2 w-48">
               <span className="text-muted-foreground whitespace-nowrap">TOKEN USAGE</span>
               <div className="flex-1 space-y-1">
                 <Progress value={45} className="h-1.5 bg-white/10" />
               </div>
               <span className="text-primary font-mono">4.2K</span>
             </div>
             <div className="h-4 w-px bg-white/10" />
             <div className="flex items-center gap-2 text-muted-foreground">
               <span>SESSION COST:</span>
               <span className="text-green-400">$0.042</span>
             </div>
          </div>

          <Sheet>
            <SheetTrigger asChild>
              <button className="flex items-center gap-2 hover:bg-white/5 px-3 py-1.5 rounded transition-colors text-muted-foreground hover:text-white border border-transparent hover:border-white/10">
                <ListTodo size={14} />
                <span>TASKS</span>
                <Badge variant="secondary" className="h-4 px-1 text-[9px] bg-primary/20 text-primary border-primary/20">3</Badge>
              </button>
            </SheetTrigger>
            <SheetContent side="right" className="bg-black/95 border-l border-white/10 backdrop-blur-xl w-80 p-0 text-white">
              <SheetHeader className="p-4 border-b border-white/10 bg-white/5">
                <SheetTitle className="text-sm font-bold flex items-center gap-2 text-white">
                  <ListTodo size={16} className="text-primary" /> MISSION OBJECTIVES
                </SheetTitle>
              </SheetHeader>
              <div className="p-4 space-y-4">
                 <div className="space-y-2">
                   <div className="text-[10px] font-bold text-muted-foreground mb-2">CURRENT PRIORITY</div>
                   
                   <div className="bg-primary/10 border border-primary/20 rounded p-3 space-y-2">
                     <div className="flex items-start gap-2">
                       <Checkbox id="task-1" className="mt-0.5 border-primary/50 data-[state=checked]:bg-primary data-[state=checked]:text-black" />
                       <div className="space-y-1">
                         <label htmlFor="task-1" className="text-xs font-bold leading-none text-white">Calibrate Neural Weights</label>
                         <p className="text-[10px] text-muted-foreground">Run optimization cycle on DGX cluster for Llama-3 endpoints.</p>
                       </div>
                     </div>
                     <div className="flex items-center gap-1 text-[9px] text-primary/70 bg-primary/5 px-2 py-1 rounded w-fit">
                       <Clock size={10} /> IN PROGRESS
                     </div>
                   </div>
                 </div>

                 <div className="space-y-2">
                   <div className="text-[10px] font-bold text-muted-foreground mb-2">BACKLOG</div>
                   
                   {[
                     { id: "task-2", text: "Integrate Vector DB", desc: "Connect Pinecone to RAG pipeline", status: "PENDING" },
                     { id: "task-3", text: "Update API Keys", desc: "Rotate OpenAI secrets in admin panel", status: "PENDING" },
                     { id: "task-4", text: "Refactor Agent Logic", desc: "Optimize token usage for chat agent", status: "DONE" }
                   ].map((task) => (
                     <div key={task.id} className={`bg-white/5 border border-white/10 rounded p-3 space-y-2 ${task.status === 'DONE' ? 'opacity-50' : ''}`}>
                       <div className="flex items-start gap-2">
                         <Checkbox id={task.id} defaultChecked={task.status === 'DONE'} className="mt-0.5 border-white/20 data-[state=checked]:bg-white/50 data-[state=checked]:border-transparent" />
                         <div className="space-y-1">
                           <label htmlFor={task.id} className={`text-xs font-bold leading-none ${task.status === 'DONE' ? 'line-through text-muted-foreground' : 'text-white'}`}>{task.text}</label>
                           <p className="text-[10px] text-muted-foreground">{task.desc}</p>
                         </div>
                       </div>
                     </div>
                   ))}
                 </div>

                 <button className="w-full flex items-center justify-center gap-2 py-2 rounded border border-dashed border-white/20 hover:bg-white/5 text-xs text-muted-foreground hover:text-white transition-colors">
                   <Plus size={14} /> ADD NEW TASK
                 </button>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>

      {/* Sidebar / Status Panel */}
      <div className="w-full md:w-80 bg-card/10 border-l border-border/50 h-screen flex flex-col p-4 gap-6 overflow-y-auto relative z-10 backdrop-blur-sm">
        
        {/* Core View Selector */}
        <div className="space-y-2">
           <div className="flex items-center justify-between">
            <h2 className="text-xs font-bold text-muted-foreground flex items-center gap-2">
              <Cpu size={14} /> 
              {selectedCoreView === 'core' && 'SYSTEM CORE'}
              {selectedCoreView === 'spark' && 'SPARKPLUG (DGX)'}
              {selectedCoreView === 'ml' && 'AI/ML WORKLOADS'}
            </h2>
            <Select value={selectedCoreView} onValueChange={setSelectedCoreView}>
              <SelectTrigger className="h-6 w-[130px] text-[10px] bg-black/40 border-primary/20 text-primary">
                <SelectValue placeholder="View" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="core">System Core</SelectItem>
                <SelectItem value="spark">SparkPlug (DGX)</SelectItem>
                <SelectItem value="ml">AI/ML Work</SelectItem>
                <SelectItem value="custom">Custom View</SelectItem>
              </SelectContent>
            </Select>
           </div>
          
          <div className="relative min-h-[200px] flex flex-col">
            <AnimatePresence mode="wait">
              {selectedCoreView === "core" && (
                <motion.div 
                  key="core"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="relative"
                >
                  <SphereSpinner isActive={isProcessing} />
                  <div className="absolute -bottom-2 w-full text-center text-[10px] text-primary/60 font-mono">
                    NEXUS KERNEL ACTIVE
                  </div>
                </motion.div>
              )}

              {selectedCoreView === "spark" && (
                <motion.div 
                  key="spark"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-3 p-3 rounded bg-black/20 border border-green-500/10 h-full"
                >
                  <div className="flex items-center justify-between border-b border-green-500/20 pb-2 mb-2">
                     <span className="text-green-400 font-bold text-xs flex items-center gap-2">
                       <Zap size={12} /> NVIDIA DGX A100
                     </span>
                     <Badge variant="outline" className="text-[10px] h-4 border-green-500/40 text-green-400 bg-green-500/5">ONLINE</Badge>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-[10px] text-muted-foreground">
                      <span>GPU CLUSTER</span>
                      <span className="text-green-300">8x A100 80GB</span>
                    </div>
                    <div className="w-full bg-black/40 h-1.5 rounded-full overflow-hidden">
                       <div className="bg-green-500 h-full w-[12%] animate-pulse" />
                    </div>
                  </div>

                  <div className="space-y-1">
                     <div className="text-[10px] font-bold text-muted-foreground flex items-center gap-1">
                       <HardDrive size={10} /> FILESYSTEM MOUNTS
                     </div>
                     <div className="grid grid-cols-1 gap-1">
                       {['/spark/datasets', '/spark/models', '/spark/checkpoints'].map(path => (
                         <div key={path} className="flex items-center gap-2 text-[10px] font-mono text-green-400/70 bg-green-500/5 p-1 rounded">
                           <FolderOpen size={10} /> {path}
                         </div>
                       ))}
                     </div>
                  </div>

                  <Dialog>
                    <DialogTrigger asChild>
                      <button className="w-full mt-2 bg-green-500/10 hover:bg-green-500/20 text-green-400 border border-green-500/30 rounded p-2 text-[10px] font-bold flex items-center justify-center gap-2 transition-colors">
                        <Terminal size={12} /> OPEN DGX COMMANDER
                      </button>
                    </DialogTrigger>
                    <DialogContent className="bg-black/95 border-green-500/30 text-green-400 max-w-4xl h-[600px] flex flex-col p-0 gap-0 font-mono backdrop-blur-xl">
                      <DialogHeader className="px-4 py-2 border-b border-green-500/20 bg-green-500/5 flex flex-row items-center justify-between">
                        <DialogTitle className="text-sm font-bold flex items-center gap-2">
                          <Zap size={16} /> DGX A100 :: SSH://192.168.1.108
                        </DialogTitle>
                        <div className="text-[10px] bg-green-500/20 px-2 py-0.5 rounded text-green-300">CONNECTED</div>
                      </DialogHeader>
                      
                      <div className="flex-1 flex overflow-hidden">
                        {/* Sidebar */}
                        <div className="w-48 border-r border-green-500/20 p-2 bg-black/40">
                          <div className="text-[10px] text-muted-foreground mb-2 px-2">MOUNT POINTS</div>
                          <div className="space-y-1">
                            {['/', '/home', '/mnt/data', '/opt/nvidia', '/tmp'].map(dir => (
                              <div key={dir} className="px-2 py-1 rounded hover:bg-green-500/10 cursor-pointer text-xs flex items-center gap-2 transition-colors">
                                <HardDrive size={12} className="opacity-70" /> {dir}
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* File List */}
                        <div className="flex-1 flex flex-col bg-black/20">
                          <div className="h-8 border-b border-green-500/20 flex items-center px-4 gap-2 text-xs bg-green-500/5">
                            <span className="text-muted-foreground">/mnt/data/</span>
                            <span className="text-white">projects/</span>
                            <span className="text-white font-bold">genesis-core/</span>
                          </div>
                          
                          <ScrollArea className="flex-1 p-2">
                             <div className="grid grid-cols-1 gap-1">
                                <div className="grid grid-cols-12 px-2 py-1 text-[10px] text-green-500/50 font-bold border-b border-green-500/10 mb-1">
                                  <div className="col-span-6">NAME</div>
                                  <div className="col-span-2">SIZE</div>
                                  <div className="col-span-2">TYPE</div>
                                  <div className="col-span-2">MODIFIED</div>
                                </div>
                                {[
                                  { name: "..", size: "", type: "DIR", mod: "" },
                                  { name: "checkpoints", size: "40 GB", type: "DIR", mod: "Today" },
                                  { name: "training_logs", size: "1.2 GB", type: "DIR", mod: "Yesterday" },
                                  { name: "config.yaml", size: "2 KB", type: "YAML", mod: "2h ago" },
                                  { name: "train.py", size: "14 KB", type: "PY", mod: "5h ago" },
                                  { name: "model_weights.pt", size: "12 GB", type: "BIN", mod: "1d ago" },
                                  { name: "dataset_manifest.json", size: "45 MB", type: "JSON", mod: "1w ago" },
                                ].map((file, i) => (
                                  <div key={i} className="grid grid-cols-12 px-2 py-1.5 rounded hover:bg-green-500/10 cursor-pointer text-xs transition-colors group items-center">
                                    <div className="col-span-6 flex items-center gap-2 text-green-300 group-hover:text-green-200">
                                      {file.type === 'DIR' ? <FolderOpen size={14} className="text-blue-400" /> : <FileText size={14} className="opacity-70" />}
                                      {file.name}
                                    </div>
                                    <div className="col-span-2 text-muted-foreground">{file.size}</div>
                                    <div className="col-span-2 text-muted-foreground">{file.type}</div>
                                    <div className="col-span-2 text-muted-foreground">{file.mod}</div>
                                  </div>
                                ))}
                             </div>
                          </ScrollArea>
                        </div>
                      </div>

                      {/* Footer Actions */}
                      <div className="h-10 border-t border-green-500/20 bg-black/60 flex items-center justify-between px-4">
                        <div className="text-[10px] text-muted-foreground flex gap-4">
                          <span>TOTAL: 54.2 GB</span>
                          <span>FREE: 1.4 TB</span>
                        </div>
                        <div className="flex gap-2">
                           <button className="flex items-center gap-1 bg-green-500/10 hover:bg-green-500/20 text-green-400 px-3 py-1 rounded text-[10px] border border-green-500/20 transition-colors">
                             <Upload size={10} /> UPLOAD
                           </button>
                           <button className="flex items-center gap-1 bg-green-500/10 hover:bg-green-500/20 text-green-400 px-3 py-1 rounded text-[10px] border border-green-500/20 transition-colors">
                             <Download size={10} /> DOWNLOAD
                           </button>
                        </div>
                      </div>
                    </DialogContent>
                  </Dialog>
                </motion.div>
              )}

              {selectedCoreView === "ml" && (
                <motion.div 
                  key="ml"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-3 p-3 rounded bg-black/20 border border-purple-500/10 h-full"
                >
                  <div className="flex items-center justify-between border-b border-purple-500/20 pb-2 mb-2">
                     <span className="text-purple-400 font-bold text-xs flex items-center gap-2">
                       <Brain size={12} /> ACTIVE WORKLOADS
                     </span>
                     <span className="text-[10px] text-purple-400/60">2 RUNNING</span>
                  </div>

                  <div className="space-y-2">
                    <div className="p-2 bg-purple-500/5 border border-purple-500/10 rounded">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-[10px] font-bold text-purple-300">TRAINING: LLAMA-3-FT</span>
                        <span className="text-[10px] text-purple-400">EPOCH 4/10</span>
                      </div>
                      <div className="w-full bg-black/40 h-1 rounded-full overflow-hidden">
                         <div className="bg-purple-500 h-full w-[45%]" />
                      </div>
                    </div>

                    <div className="p-2 bg-purple-500/5 border border-purple-500/10 rounded">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-[10px] font-bold text-purple-300">INFERENCE: RAG-PIPELINE</span>
                        <span className="text-[10px] text-green-400 animate-pulse">SERVING</span>
                      </div>
                      <div className="flex gap-2 text-[10px] text-muted-foreground font-mono mt-1">
                         <span className="flex items-center gap-1"><Activity size={8}/> 45 req/s</span>
                         <span className="flex items-center gap-1"><FileCode size={8}/> 12ms</span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
              {selectedCoreView === "custom" && (
                <motion.div 
                  key="custom"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-3 p-3 rounded bg-black/20 border border-blue-500/10 h-full flex flex-col"
                >
                  <div className="flex items-center justify-between border-b border-blue-500/20 pb-2 mb-2">
                     <span className="text-blue-400 font-bold text-xs flex items-center gap-2">
                       <Braces size={12} /> CUSTOM CONFIG
                     </span>
                     <Badge variant="outline" className="text-[10px] h-4 border-blue-500/40 text-blue-400 bg-blue-500/5">JSON</Badge>
                  </div>
                  
                  <div className="flex-1 overflow-hidden relative group">
                    <div className="absolute inset-0 bg-black/40 rounded border border-white/5 p-2 font-mono text-[10px] text-muted-foreground overflow-auto">
                      <span className="text-pink-400">{"{"}</span><br/>
                      &nbsp;&nbsp;<span className="text-blue-300">"custom_mcp"</span>: <span className="text-pink-400">{"["}</span><br/>
                      &nbsp;&nbsp;&nbsp;&nbsp;<span className="text-pink-400">{"{"}</span><br/>
                      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="text-blue-300">"id"</span>: <span className="text-green-300">"weather-api"</span>,<br/>
                      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="text-blue-300">"status"</span>: <span className="text-green-300">"active"</span><br/>
                      &nbsp;&nbsp;&nbsp;&nbsp;<span className="text-pink-400">{"}"}</span>,<br/>
                      &nbsp;&nbsp;&nbsp;&nbsp;<span className="text-pink-400">{"{"}</span><br/>
                      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="text-blue-300">"id"</span>: <span className="text-green-300">"trading-bot"</span>,<br/>
                      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="text-blue-300">"status"</span>: <span className="text-yellow-300">"idle"</span><br/>
                      &nbsp;&nbsp;&nbsp;&nbsp;<span className="text-pink-400">{"}"}</span><br/>
                      &nbsp;&nbsp;<span className="text-pink-400">{"]"}</span><br/>
                      <span className="text-pink-400">{"}"}</span>
                    </div>
                  </div>

                  <div className="space-y-1 mt-auto">
                     <div className="text-[10px] font-bold text-muted-foreground flex items-center gap-1">
                       <Database size={10} /> LINKED RESOURCES
                     </div>
                     <div className="grid grid-cols-2 gap-2">
                         <div className="flex items-center gap-1 text-[9px] font-mono text-blue-400/70 bg-blue-500/5 p-1 rounded">
                           <div className="w-1.5 h-1.5 rounded-full bg-green-500"></div> Weather
                         </div>
                         <div className="flex items-center gap-1 text-[9px] font-mono text-blue-400/70 bg-blue-500/5 p-1 rounded">
                           <div className="w-1.5 h-1.5 rounded-full bg-yellow-500"></div> Trading
                         </div>
                     </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
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
