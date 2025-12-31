import { motion, AnimatePresence } from "framer-motion";
import { User, Settings, Users, Palette, Shield, ChevronRight, LayoutGrid, FileCode, Wrench, BrainCircuit, Hammer, FolderCog, Sparkles, Folder, Plug, Server, Box, Globe, Database, Briefcase, Plus, Activity, Github, Edit3, GitBranch, Bot, Factory, ShieldCheck, Check } from "lucide-react";
import { useLocation } from "wouter";
import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription, SheetTrigger } from "@/components/ui/sheet";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export default function Admin() {
  const [_, setLocation] = useLocation();
  const [activeTab, setActiveTab] = useState("profile");

  const menuItems = [
    { 
      id: "profile", 
      title: "User Profile", 
      icon: <User size={18} />,
      description: "Manage personal settings and credentials"
    },
    { 
      id: "projects", 
      title: "Project Profiles", 
      icon: <Briefcase size={18} />,
      description: "Manage workspaces and project-specific configurations"
    },
    { 
      id: "systems", 
      title: "Systems & Configurations", 
      icon: <Settings size={18} />,
      description: "Manage APIs, integrations, and runtime environment"
    },
    { 
      id: "agents", 
      title: "Agent Management", 
      icon: <Users size={18} />,
      description: "Orchestrate active agents and resource allocation"
    },
    { 
      id: "foundry", 
      title: "Agent Foundry", 
      icon: <BrainCircuit size={18} />,
      description: "Advanced skill creation and tool allocation"
    },
    { 
      id: "customization", 
      title: "Customizations", 
      icon: <Palette size={18} />,
      description: "Theme editor and UI preferences"
    }
  ];

  const tools = [
    { id: "fs", name: "FileSystem Access", category: "Core", icon: <FolderCog size={14} /> },
    { id: "web", name: "Web Browser", category: "Network", icon: <LayoutGrid size={14} /> },
    { id: "code", name: "Code Interpreter", category: "Compute", icon: <FileCode size={14} /> },
    { id: "img", name: "Image Gen", category: "Creative", icon: <Palette size={14} /> },
  ];

  return (
    <div className="min-h-screen bg-background text-foreground font-mono overflow-hidden flex flex-col relative">
      <div className="scanline" />
      <div className="pointer-events-none fixed inset-0 z-50 bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.4)_100%)]" />

      {/* Header */}
      <header className="h-16 border-b border-border/50 bg-card/20 flex items-center justify-between px-6 backdrop-blur-md relative z-10">
        <div className="flex items-center gap-3 text-primary cursor-pointer hover:opacity-80 transition-opacity" onClick={() => setLocation("/")}>
          <Shield size={20} />
          <h1 className="font-display font-bold tracking-widest text-xl">SPARKPLUG ADMIN</h1>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="text-xs font-mono text-muted-foreground flex items-center gap-2">
            <span>PROJECT =</span>
            <span className="text-primary font-bold flex items-center gap-2">
              <Box size={14} /> SparkPlug DGX
            </span>
          </div>

          <Select defaultValue="main">
            <SelectTrigger className="w-[200px] h-8 text-xs font-mono bg-black/40 border-white/10 text-muted-foreground">
               <SelectValue placeholder="Select Sub-Project" />
            </SelectTrigger>
            <SelectContent className="bg-black/90 border-white/10 backdrop-blur-xl">
               <SelectItem value="main">Main Cluster (Default)</SelectItem>
               <SelectItem value="research">Research Node Alpha</SelectItem>
               <SelectItem value="web">Web Services Delta</SelectItem>
               <SelectItem value="data">Data Lake Omega</SelectItem>
            </SelectContent>
          </Select>

          <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground bg-black/40 px-3 py-1.5 rounded border border-white/5">
            <div className="w-2 h-2 rounded-full bg-orange-500 animate-pulse" />
            ADMIN MODE
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden flex relative z-10 max-w-7xl mx-auto w-full p-6 gap-6">
        <Tabs defaultValue="profile" value={activeTab} onValueChange={setActiveTab} orientation="vertical" className="flex w-full h-full gap-6">
          
          {/* Sidebar Navigation */}
          <div className="w-64 shrink-0 flex flex-col gap-2">
            <div className="mb-4 px-2">
              <h2 className="text-xs font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                <LayoutGrid size={14} /> Modules
              </h2>
            </div>
            
            <TabsList className="flex flex-col h-auto bg-transparent gap-2 p-0">
              {menuItems.map((item) => (
                <TabsTrigger
                  key={item.id}
                  value={item.id}
                  className="w-full justify-start gap-3 px-4 py-3 h-auto border border-transparent data-[state=active]:bg-card/40 data-[state=active]:border-primary/30 data-[state=active]:text-primary transition-all duration-300 font-mono text-sm group"
                >
                  <div className={`p-1.5 rounded bg-black/20 group-data-[state=active]:text-primary text-muted-foreground transition-colors`}>
                    {item.icon}
                  </div>
                  <div className="flex flex-col items-start text-left">
                    <span className="font-bold">{item.title}</span>
                  </div>
                  <ChevronRight className="ml-auto opacity-0 group-data-[state=active]:opacity-100 transition-opacity" size={14} />
                </TabsTrigger>
              ))}
            </TabsList>
          </div>

          {/* Content Area */}
          <div className="flex-1 h-full overflow-y-auto pr-2">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
                className="h-full"
              >
                <TabsContent value="profile" className="mt-0 h-full border-none p-0">
                  <div className="space-y-6">
                    <div>
                      <h2 className="text-2xl font-display font-bold text-primary mb-1">User Profile</h2>
                      <p className="text-muted-foreground">Manage your identity and security credentials.</p>
                    </div>
                    
                    <Card className="bg-card/30 border-border/50 backdrop-blur-sm">
                      <CardHeader>
                        <CardTitle className="font-mono text-lg">Personal Information</CardTitle>
                        <CardDescription>Update your display name and contact details.</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label htmlFor="username">Username</Label>
                            <Input id="username" defaultValue="Administrator" className="bg-black/20 border-border/50 font-mono" />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="email">Email</Label>
                            <Input id="email" defaultValue="admin@nexus-cli.dev" className="bg-black/20 border-border/50 font-mono" />
                          </div>
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="bio">Bio</Label>
                          <Input id="bio" defaultValue="System Architect & AI Operator" className="bg-black/20 border-border/50 font-mono" />
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-card/30 border-border/50 backdrop-blur-sm">
                      <CardHeader>
                        <CardTitle className="font-mono text-lg text-destructive">Danger Zone</CardTitle>
                        <CardDescription>Irreversible actions for your account.</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <Button variant="destructive" className="font-mono">DELETE ACCOUNT</Button>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>

                <TabsContent value="projects" className="mt-0 h-full border-none p-0">
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <h2 className="text-2xl font-display font-bold text-blue-400 mb-1">Project Profiles</h2>
                        <p className="text-muted-foreground">Manage and switch between different project workspaces.</p>
                      </div>
                      
                      <Dialog>
                        <DialogTrigger asChild>
                          <Button className="bg-blue-600 hover:bg-blue-700 text-xs font-mono gap-2">
                            <Plus size={14} /> NEW PROJECT
                          </Button>
                        </DialogTrigger>
                        <DialogContent className="bg-black/90 border-blue-500/20 text-white backdrop-blur-xl">
                          <DialogHeader>
                            <DialogTitle className="font-mono text-xl text-blue-400 flex items-center gap-2">
                              <Box size={20} /> INITIALIZE PROJECT
                            </DialogTitle>
                            <DialogDescription className="text-gray-400">
                              Configure a new workspace environment.
                            </DialogDescription>
                          </DialogHeader>
                          <div className="space-y-4 py-4">
                            <div className="space-y-2">
                              <Label className="text-xs font-mono text-blue-300">PROJECT CODENAME</Label>
                              <Input placeholder="e.g. OMEGA-PROTOCOL" className="bg-blue-900/10 border-blue-500/30 font-mono text-white placeholder:text-blue-500/30" />
                            </div>
                            <div className="space-y-2">
                              <Label className="text-xs font-mono text-blue-300">WORKLOAD TYPE</Label>
                              <div className="grid grid-cols-2 gap-3">
                                <div className="p-3 rounded border border-blue-500/50 bg-blue-500/10 cursor-pointer hover:bg-blue-500/20 transition-colors">
                                  <div className="font-bold text-xs mb-1">AI / ML RESEARCH</div>
                                  <div className="text-[10px] text-gray-400">Includes DGX SparkPlug & Foundry</div>
                                </div>
                                <div className="p-3 rounded border border-white/10 bg-white/5 cursor-pointer hover:bg-white/10 transition-colors opacity-60">
                                  <div className="font-bold text-xs mb-1">WEB APPLICATION</div>
                                  <div className="text-[10px] text-gray-400">Standard Full-Stack Env</div>
                                </div>
                                <div className="p-3 rounded border border-white/10 bg-white/5 cursor-pointer hover:bg-white/10 transition-colors opacity-60">
                                  <div className="font-bold text-xs mb-1 flex items-center gap-2"><Github size={12} /> CONNECT GITHUB</div>
                                  <div className="text-[10px] text-gray-400">Import Repository</div>
                                </div>
                                <div className="p-3 rounded border border-white/10 bg-white/5 cursor-pointer hover:bg-white/10 transition-colors opacity-60">
                                  <div className="font-bold text-xs mb-1 flex items-center gap-2"><Edit3 size={12} /> CREATE CUSTOM</div>
                                  <div className="text-[10px] text-gray-400">Manual Configuration</div>
                                </div>
                              </div>
                            </div>
                          </div>
                          <DialogFooter>
                            <Button variant="outline" className="border-white/10 hover:bg-white/5 text-gray-400">CANCEL</Button>
                            <Button className="bg-blue-600 hover:bg-blue-500 text-white font-mono">
                              <Sparkles size={14} className="mr-2" /> INITIALIZE
                            </Button>
                          </DialogFooter>
                        </DialogContent>
                      </Dialog>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      <Card className="bg-blue-500/10 border-blue-500/50 backdrop-blur-sm relative overflow-hidden group cursor-pointer hover:bg-blue-500/20 transition-all">
                        <div className="absolute top-0 right-0 p-2">
                          <Badge className="bg-blue-500 hover:bg-blue-600 text-white font-mono text-[10px]">ACTIVE</Badge>
                        </div>
                        <CardHeader>
                          <CardTitle className="font-mono text-lg flex items-center gap-2">
                            <BrainCircuit size={18} className="text-blue-400" /> Genesis
                          </CardTitle>
                          <CardDescription>DGX/AIML Research Core</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                          <div className="space-y-2 text-xs font-mono text-muted-foreground">
                            <div className="flex justify-between"><span>Type:</span> <span className="text-foreground">AI Research</span></div>
                            <div className="flex justify-between"><span>Modules:</span> <span className="text-foreground">SparkPlug, Foundry</span></div>
                            <div className="flex justify-between"><span>Created:</span> <span className="text-foreground">2024-12-01</span></div>
                          </div>
                          <Separator className="bg-blue-500/20" />
                          <div className="flex gap-2">
                            <Button size="sm" variant="outline" className="w-full text-xs h-7 border-blue-500/30 hover:bg-blue-500/20 hover:text-blue-400">CONFIGURE</Button>
                          </div>
                        </CardContent>
                      </Card>

                      <Card className="bg-card/30 border-border/50 backdrop-blur-sm relative overflow-hidden group cursor-pointer hover:border-primary/50 transition-all">
                        <CardHeader>
                          <CardTitle className="font-mono text-lg flex items-center gap-2">
                            <Globe size={18} className="text-orange-400" /> Helios Web
                          </CardTitle>
                          <CardDescription>Full-Stack Web Application</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                          <div className="space-y-2 text-xs font-mono text-muted-foreground">
                            <div className="flex justify-between"><span>Type:</span> <span className="text-foreground">Web Dev</span></div>
                            <div className="flex justify-between"><span>Modules:</span> <span className="text-foreground">React, Node.js</span></div>
                            <div className="flex justify-between"><span>Created:</span> <span className="text-foreground">2025-01-15</span></div>
                          </div>
                          <Separator className="bg-border/30" />
                          <div className="flex gap-2">
                            <Button size="sm" variant="outline" className="w-full text-xs h-7">ACTIVATE</Button>
                            <Button size="sm" variant="outline" className="w-full text-xs h-7">SETTINGS</Button>
                          </div>
                        </CardContent>
                      </Card>

                      <Card className="bg-card/10 border-dashed border-border flex flex-col items-center justify-center min-h-[200px] cursor-pointer hover:bg-card/20 hover:border-primary/30 transition-all group">
                        <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                          <Plus size={24} className="text-primary" />
                        </div>
                        <h3 className="font-mono font-bold text-sm">Create Profile</h3>
                        <p className="text-xs text-muted-foreground mt-1">Start a new workspace</p>
                      </Card>
                    </div>

                    <Card className="bg-card/30 border-border/50 backdrop-blur-sm">
                      <CardHeader>
                        <CardTitle className="font-mono text-lg">Profile Customization</CardTitle>
                        <CardDescription>Override global defaults for the active project.</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="space-y-4">
                            <div className="space-y-2">
                              <Label>Interface Theme</Label>
                              <div className="grid grid-cols-3 gap-2">
                                <div className="h-8 rounded bg-cyan-500/20 border border-cyan-500/50 cursor-pointer" title="Cyberpunk (Default)" />
                                <div className="h-8 rounded bg-orange-500/20 border border-orange-500/20 cursor-pointer hover:border-orange-500/50" title="Sunset" />
                                <div className="h-8 rounded bg-purple-500/20 border border-purple-500/20 cursor-pointer hover:border-purple-500/50" title="Void" />
                              </div>
                            </div>
                            <div className="space-y-2">
                              <Label>Enabled Modules</Label>
                              <div className="space-y-2">
                                <div className="flex items-center justify-between p-2 rounded bg-black/20 border border-white/5">
                                  <span className="text-xs font-mono">Agent Foundry</span>
                                  <Switch defaultChecked />
                                </div>
                                <div className="flex items-center justify-between p-2 rounded bg-black/20 border border-white/5">
                                  <span className="text-xs font-mono">SparkPlug (DGX)</span>
                                  <Switch defaultChecked />
                                </div>
                                <div className="flex items-center justify-between p-2 rounded bg-black/20 border border-white/5">
                                  <span className="text-xs font-mono">Web Preview</span>
                                  <Switch />
                                </div>
                              </div>
                            </div>
                          </div>
                          
                          <div className="space-y-2">
                            <Label>Project Environment</Label>
                            <div className="h-full bg-black/40 rounded border border-white/10 p-3 font-mono text-xs text-muted-foreground">
                              # Project Specific ENV
                              <br/>
                              PROJECT_TYPE="research"
                              <br/>
                              DGX_ALLOCATION="8x"
                              <br/>
                              <br/>
                              <span className="text-green-400"># Overrides applied</span>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>

                <TabsContent value="systems" className="mt-0 h-full border-none p-0">
                  <div className="space-y-6">
                    <div>
                      <h2 className="text-2xl font-display font-bold text-secondary mb-1">Systems & Configurations</h2>
                      <p className="text-muted-foreground">Configure global variables, integrations, and runtime environments.</p>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      <div className="space-y-6">
                        <Card className="bg-card/30 border-border/50 backdrop-blur-sm">
                          <CardHeader>
                            <CardTitle className="font-mono text-lg flex items-center gap-2">
                              <BrainCircuit size={16} className="text-primary" /> Model Configuration
                            </CardTitle>
                            <CardDescription>Configure AI model providers and API access.</CardDescription>
                          </CardHeader>
                          <CardContent className="space-y-6">
                            {/* Proxy OAuth / BYOK Section */}
                            <div className="p-4 rounded border border-blue-500/20 bg-blue-500/5 space-y-3">
                               <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-2">
                                     <ShieldCheck size={16} className="text-blue-400" />
                                     <span className="font-bold text-sm text-blue-100">Proxy OAuth / BYOK</span>
                                  </div>
                                  <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20 text-[10px]">SECURE VAULT</Badge>
                               </div>
                               <p className="text-xs text-muted-foreground">
                                  Securely connect your own enterprise subscriptions or API keys. Credentials are encrypted and stored locally.
                               </p>
                            </div>

                            <Tabs defaultValue="anthropic" className="w-full">
                              <TabsList className="grid grid-cols-4 bg-black/40 h-auto p-1">
                                <TabsTrigger value="anthropic" className="text-xs font-mono py-2">Anthropic</TabsTrigger>
                                <TabsTrigger value="openai" className="text-xs font-mono py-2">OpenAI</TabsTrigger>
                                <TabsTrigger value="gemini" className="text-xs font-mono py-2">Gemini</TabsTrigger>
                                <TabsTrigger value="factory" className="text-xs font-mono py-2">Factory</TabsTrigger>
                              </TabsList>
                              
                              <TabsContent value="anthropic" className="space-y-4 mt-4">
                                <div className="space-y-2">
                                  <Label>Anthropic API Key</Label>
                                  <div className="flex gap-2">
                                    <Input type="password" placeholder="sk-ant-..." className="bg-black/20 border-border/50 font-mono flex-1" />
                                    <Button className="bg-primary/20 text-primary border border-primary/50 hover:bg-primary/30">Connect</Button>
                                  </div>
                                  <p className="text-[10px] text-muted-foreground">Required for Claude 3.5 Sonnet & Opus access.</p>
                                </div>
                              </TabsContent>

                              <TabsContent value="openai" className="space-y-4 mt-4">
                                <div className="space-y-2">
                                  <Label>OpenAI / Codex Key</Label>
                                  <div className="flex gap-2">
                                    <Input type="password" value="sk-........................" readOnly className="bg-black/20 border-border/50 font-mono flex-1" />
                                    <Button variant="outline" className="border-green-500/30 text-green-400 hover:bg-green-500/10 gap-2">
                                       <Check size={14} /> Connected
                                    </Button>
                                  </div>
                                </div>
                              </TabsContent>

                              <TabsContent value="gemini" className="space-y-4 mt-4">
                                <div className="space-y-2">
                                  <Label>Google AI Studio Key</Label>
                                  <div className="flex gap-2">
                                    <Input type="password" placeholder="AIzaSy..." className="bg-black/20 border-border/50 font-mono flex-1" />
                                    <Button className="bg-primary/20 text-primary border border-primary/50 hover:bg-primary/30">Connect</Button>
                                  </div>
                                </div>
                                <div className="flex items-center gap-2 p-2 rounded bg-yellow-500/10 border border-yellow-500/20 text-yellow-500 text-xs">
                                   <Activity size={14} />
                                   <span>Enterprise: Vertex AI Project ID required for production workloads.</span>
                                </div>
                              </TabsContent>

                              <TabsContent value="factory" className="space-y-4 mt-4">
                                <div className="space-y-2">
                                  <Label>Factory Droid Subscription</Label>
                                  <div className="flex flex-col gap-3">
                                    <Button className="w-full bg-[#ff4f00]/20 text-[#ff4f00] border border-[#ff4f00]/50 hover:bg-[#ff4f00]/30 h-10 font-bold">
                                       <Factory size={16} className="mr-2" /> AUTHENTICATE WITH FACTORY
                                    </Button>
                                    <div className="text-center text-[10px] text-muted-foreground">- OR -</div>
                                    <Input type="password" placeholder="fd_live_..." className="bg-black/20 border-border/50 font-mono" />
                                  </div>
                                </div>
                              </TabsContent>
                            </Tabs>

                            <Separator className="bg-white/10" />

                            <div className="space-y-2">
                              <Label>Local Inference Override</Label>
                              <div className="flex items-center justify-between p-3 rounded bg-black/20 border border-white/5">
                                <div className="flex items-center gap-3">
                                  <Server size={16} className="text-orange-500" />
                                  <div>
                                    <div className="font-bold text-sm">Ollama / LocalAI</div>
                                    <div className="text-[10px] text-muted-foreground">http://localhost:11434</div>
                                  </div>
                                </div>
                                <Switch />
                              </div>
                            </div>
                          </CardContent>
                        </Card>

                        <Card className="bg-card/30 border-border/50 backdrop-blur-sm">
                          <CardHeader>
                            <CardTitle className="font-mono text-lg flex items-center gap-2">
                              <Server size={16} className="text-orange-500" /> MCP Integrations
                            </CardTitle>
                            <CardDescription>Model Context Protocol servers.</CardDescription>
                          </CardHeader>
                          <CardContent className="space-y-3">
                            {[
                              { name: "PostgreSQL Connector", status: "Active", type: "Database" },
                              { name: "Filesystem Watcher", status: "Active", type: "System" },
                              { name: "GitHub Repository", status: "Inactive", type: "VCS" },
                              { name: "Memory Service", status: "Inactive", type: "Core" },
                              { name: "Google Drive", status: "Inactive", type: "Storage" }
                            ].map((mcp) => (
                              <div key={mcp.name} className="flex items-center justify-between p-2 rounded bg-black/20 border border-white/5">
                                <div className="flex items-center gap-3">
                                  <div className={`w-2 h-2 rounded-full ${mcp.status === 'Active' ? 'bg-green-500 shadow-[0_0_5px_#22c55e]' : 'bg-muted'}`} />
                                  <div>
                                    <div className="font-bold text-xs">{mcp.name}</div>
                                    <div className="text-[10px] text-muted-foreground">{mcp.type}</div>
                                  </div>
                                </div>
                                <Button size="sm" variant="ghost" className="h-6 text-[10px]">CONFIG</Button>
                              </div>
                            ))}
                            <Button variant="outline" className="w-full border-dashed border-border text-muted-foreground hover:text-primary hover:border-primary/50 h-8 text-xs">
                              + ADD MCP SERVER
                            </Button>
                          </CardContent>
                        </Card>
                      </div>

                      <div className="space-y-6">
                        <Card className="bg-card/30 border-border/50 backdrop-blur-sm">
                          <CardHeader>
                            <CardTitle className="font-mono text-lg flex items-center gap-2">
                              <Plug size={16} className="text-purple-500" /> External Tools & APIs
                            </CardTitle>
                            <CardDescription>Connected services and toolkits.</CardDescription>
                          </CardHeader>
                          <CardContent className="space-y-4">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div className="p-3 rounded bg-card/50 border border-border flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                  <div className="p-2 rounded bg-black/40 text-white"><FileCode size={16} /></div>
                                  <div>
                                    <div className="font-bold text-sm">GitHub</div>
                                    <div className="text-[10px] text-muted-foreground">Code & Issues</div>
                                  </div>
                                </div>
                                <Switch />
                              </div>

                              <div className="p-3 rounded bg-card/50 border border-border flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                  <div className="p-2 rounded bg-black/40 text-blue-400"><Database size={16} /></div>
                                  <div>
                                    <div className="font-bold text-sm">PostgreSQL</div>
                                    <div className="text-[10px] text-muted-foreground">Structured DB</div>
                                  </div>
                                </div>
                                <Switch defaultChecked />
                              </div>

                              <div className="p-3 rounded bg-card/50 border border-border flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                  <div className="p-2 rounded bg-black/40 text-yellow-400"><Database size={16} /></div>
                                  <div>
                                    <div className="font-bold text-sm">Pinecone</div>
                                    <div className="text-[10px] text-green-400">Connected</div>
                                  </div>
                                </div>
                                <Switch defaultChecked />
                              </div>

                              <div className="p-3 rounded bg-card/50 border border-border flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                  <div className="p-2 rounded bg-black/40 text-orange-400"><Globe size={16} /></div>
                                  <div>
                                    <div className="font-bold text-sm">Brave Search</div>
                                    <div className="text-[10px] text-green-400">Connected</div>
                                  </div>
                                </div>
                                <Switch defaultChecked />
                              </div>

                              <div className="p-3 rounded bg-card/50 border border-border flex items-center justify-between opacity-60">
                                <div className="flex items-center gap-3">
                                  <div className="p-2 rounded bg-black/40 text-pink-400"><Box size={16} /></div>
                                  <div>
                                    <div className="font-bold text-sm">Slack</div>
                                    <div className="text-[10px] text-muted-foreground">Webhooks</div>
                                  </div>
                                </div>
                                <Switch />
                              </div>

                              <div className="p-3 rounded bg-card/50 border border-border flex items-center justify-between opacity-60">
                                <div className="flex items-center gap-3">
                                  <div className="p-2 rounded bg-black/40 text-white"><LayoutGrid size={16} /></div>
                                  <div>
                                    <div className="font-bold text-sm">Vercel</div>
                                    <div className="text-[10px] text-muted-foreground">Deployment</div>
                                  </div>
                                </div>
                                <Switch />
                              </div>
                              
                              <div className="p-3 rounded bg-card/50 border border-border flex items-center justify-between opacity-60">
                                <div className="flex items-center gap-3">
                                  <div className="p-2 rounded bg-black/40 text-purple-400"><Activity size={16} /></div>
                                  <div>
                                    <div className="font-bold text-sm">Sentry</div>
                                    <div className="text-[10px] text-muted-foreground">Monitoring</div>
                                  </div>
                                </div>
                                <Switch />
                              </div>

                              <div className="p-3 rounded bg-card/50 border border-border flex items-center justify-between opacity-60">
                                <div className="flex items-center gap-3">
                                  <div className="p-2 rounded bg-black/40 text-yellow-600"><BrainCircuit size={16} /></div>
                                  <div>
                                    <div className="font-bold text-sm">Hugging Face</div>
                                    <div className="text-[10px] text-muted-foreground">Models</div>
                                  </div>
                                </div>
                                <Switch />
                              </div>
                            </div>

                            <Button variant="outline" className="w-full border-dashed border-border text-muted-foreground hover:text-primary hover:border-primary/50 h-10 text-xs mt-4">
                              <Plus size={14} className="mr-2" /> ADD CUSTOM TOOL / API
                            </Button>
                          </CardContent>
                        </Card>

                        <Card className="bg-card/30 border-border/50 backdrop-blur-sm">
                          <CardHeader>
                            <CardTitle className="font-mono text-lg">Build Pipeline</CardTitle>
                          </CardHeader>
                          <CardContent className="space-y-4">
                            <div className="flex items-center justify-between">
                              <div className="space-y-0.5">
                                <Label>Auto-Deploy Agents</Label>
                                <p className="text-xs text-muted-foreground">Automatically deploy agents after successful build</p>
                              </div>
                              <Switch />
                            </div>
                            <Separator className="bg-border/30" />
                            <div className="flex items-center justify-between">
                              <div className="space-y-0.5">
                                <Label>Verbose Logging</Label>
                                <p className="text-xs text-muted-foreground">Enable detailed build logs in terminal</p>
                              </div>
                              <Switch defaultChecked />
                            </div>
                          </CardContent>
                        </Card>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="agents" className="mt-0 h-full border-none p-0">
                  <div className="space-y-6">
                    <div>
                      <h2 className="text-2xl font-display font-bold text-green-400 mb-1">Agent Management</h2>
                      <p className="text-muted-foreground">Monitor and orchestrate your autonomous workforce.</p>
                    </div>

                    <div className="grid grid-cols-1 gap-6">
                      {[{
                        id: "CODER-ALPHA",
                        role: "Lead Architect",
                        status: "active",
                        uptime: "42h 12m",
                        subAgents: [
                           { id: "CODE-GEN-01", role: "Frontend Implementation", status: "active", uptime: "2h 45m" },
                           { id: "TEST-BOT-X", role: "Unit Testing", status: "idle", uptime: "12h 10m" }
                        ]
                      }, {
                        id: "SECURITY-PRIME",
                        role: "Systems Overwatch",
                        status: "active",
                        uptime: "156h 30m",
                        subAgents: [
                           { id: "AUDIT-LOG-V2", role: "Log Analysis", status: "active", uptime: "156h 30m" }
                        ]
                      }, {
                        id: "DATA-SENTRY",
                        role: "Knowledge Manager",
                        status: "active",
                        uptime: "8h 15m",
                        subAgents: []
                      }].map((agent) => (
                        <Card key={agent.id} className="bg-card/30 border-border/50 backdrop-blur-sm overflow-hidden relative">
                          <CardHeader className="pb-2 border-b border-white/5 bg-white/5">
                             <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                   <div className="w-10 h-10 rounded bg-primary/20 flex items-center justify-center border border-primary/30">
                                      <Bot size={20} className="text-primary" />
                                   </div>
                                   <div>
                                      <CardTitle className="font-mono text-md flex items-center gap-2">
                                         {agent.id}
                                         <Badge variant="outline" className="text-[10px] h-5 border-green-500/30 text-green-400 bg-green-500/10">ONLINE</Badge>
                                      </CardTitle>
                                      <CardDescription className="text-xs">{agent.role}</CardDescription>
                                   </div>
                                </div>
                                <div className="text-right">
                                   <div className="text-[10px] text-muted-foreground font-mono">UPTIME</div>
                                   <div className="text-xs font-bold text-white font-mono">{agent.uptime}</div>
                                </div>
                             </div>
                          </CardHeader>
                          <CardContent className="pt-4 space-y-4">
                            {/* Main Agent Stats */}
                            <div className="space-y-2 text-xs font-mono">
                              <div className="flex justify-between">
                                <span>Compute Load</span>
                                <span className="text-primary">32%</span>
                              </div>
                              <div className="w-full bg-black/40 h-1.5 rounded-full overflow-hidden">
                                <div className="bg-primary/50 h-full w-[32%]" />
                              </div>
                            </div>
                            
                            {/* Sub Agents Hierarchy */}
                            <div className="space-y-3 pl-4 border-l border-white/10 ml-4 relative">
                               <div className="absolute -left-[17px] top-0 bottom-0 w-px bg-white/10" />
                               
                               <div className="text-[10px] font-bold text-muted-foreground flex items-center gap-2">
                                  <GitBranch size={12} className="rotate-90" /> SUB-AGENTS ({agent.subAgents.length})
                               </div>

                               {agent.subAgents.length > 0 ? (
                                 <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                    {agent.subAgents.map(sub => (
                                       <div key={sub.id} className="bg-black/20 border border-white/5 rounded p-2 flex items-center justify-between">
                                          <div className="flex items-center gap-2">
                                             <div className={`w-1.5 h-1.5 rounded-full ${sub.status === 'active' ? 'bg-green-500' : 'bg-yellow-500'} animate-pulse`} />
                                             <div>
                                                <div className="font-bold text-xs text-white">{sub.id}</div>
                                                <div className="text-[10px] text-muted-foreground">{sub.role}</div>
                                             </div>
                                          </div>
                                          <Sheet>
                                            <SheetTrigger asChild>
                                              <Button size="sm" variant="ghost" className="h-6 w-6 p-0 hover:bg-white/10 rounded-full">
                                                 <Settings size={12} className="text-muted-foreground" />
                                              </Button>
                                            </SheetTrigger>
                                            <SheetContent className="bg-black/90 border-l border-white/10 backdrop-blur-xl min-w-[400px]">
                                              <SheetHeader>
                                                <SheetTitle className="font-mono text-blue-400 flex items-center gap-2">
                                                  <Bot size={18} /> AGENT CONFIG: {sub.id}
                                                </SheetTitle>
                                                <SheetDescription>
                                                  Configure behavioral parameters and capabilities.
                                                </SheetDescription>
                                              </SheetHeader>
                                              
                                              <div className="space-y-6 mt-6">
                                                {/* Instructions */}
                                                <div className="space-y-2">
                                                  <Label className="text-xs font-mono text-muted-foreground">SYSTEM INSTRUCTIONS</Label>
                                                  <Textarea 
                                                    className="bg-black/40 border-white/10 font-mono text-xs h-32 text-gray-300" 
                                                    defaultValue={`You are ${sub.role}. Your primary directive is to execute tasks with precision and minimal resource consumption.\n\nReport all anomalies to SECURITY-PRIME.`}
                                                  />
                                                </div>

                                                {/* Permissions */}
                                                <div className="space-y-3">
                                                  <Label className="text-xs font-mono text-muted-foreground">TOOL PERMISSIONS</Label>
                                                  <div className="grid grid-cols-2 gap-2">
                                                    {['Filesystem Read', 'Filesystem Write', 'Network Access', 'Process Control', 'Memory Dump', 'Code Execution'].map(perm => (
                                                      <div key={perm} className="flex items-center justify-between p-2 rounded bg-white/5 border border-white/5">
                                                        <span className="text-[10px] font-mono">{perm}</span>
                                                        <Switch defaultChecked={['Filesystem Read', 'Network Access'].includes(perm)} className="scale-75" />
                                                      </div>
                                                    ))}
                                                  </div>
                                                </div>

                                                {/* Skills */}
                                                 <div className="space-y-3">
                                                  <Label className="text-xs font-mono text-muted-foreground">ACTIVE SKILLS</Label>
                                                  <div className="space-y-2">
                                                    {['LogParser_v2.py', 'AnomalyDetector.js'].map(skill => (
                                                       <div key={skill} className="flex items-center justify-between p-2 rounded bg-purple-500/10 border border-purple-500/20">
                                                         <div className="flex items-center gap-2">
                                                           <FileCode size={12} className="text-purple-400" />
                                                           <span className="text-[10px] font-mono text-purple-200">{skill}</span>
                                                         </div>
                                                         <Button variant="ghost" size="sm" className="h-5 w-5 p-0 text-purple-400 hover:text-purple-200 hover:bg-purple-500/20">
                                                           <Settings size={10} />
                                                         </Button>
                                                       </div>
                                                    ))}
                                                    <Button variant="outline" className="w-full h-7 text-[10px] border-dashed border-white/10 text-muted-foreground hover:text-white hover:bg-white/5">
                                                      + ASSIGN SKILL
                                                    </Button>
                                                  </div>
                                                </div>
                                                
                                                {/* Actions */}
                                                <div className="pt-4 flex gap-2">
                                                   <Button className="w-full bg-blue-600 hover:bg-blue-500 text-xs font-mono">APPLY CONFIG</Button>
                                                   <Button variant="outline" className="w-full border-red-500/30 text-red-400 hover:bg-red-500/10 text-xs font-mono">RESET MEMORY</Button>
                                                </div>
                                              </div>
                                            </SheetContent>
                                          </Sheet>
                                       </div>
                                    ))}
                                 </div>
                               ) : (
                                  <div className="text-[10px] text-muted-foreground italic p-2 border border-dashed border-white/10 rounded bg-white/5">
                                     No sub-agents deployed.
                                  </div>
                               )}
                            </div>

                            <Separator className="bg-white/5" />
                            <div className="flex gap-2 justify-end">
                              <Button size="sm" variant="outline" className="h-7 text-xs border-white/10 hover:bg-white/5 text-muted-foreground">VIEW LOGS</Button>
                              <Button size="sm" variant="outline" className="h-7 text-xs border-primary/50 text-primary hover:bg-primary/10">MANAGE FLEET</Button>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                      
                      <Card className="bg-card/10 border-dashed border-border flex items-center justify-center min-h-[100px] cursor-pointer hover:bg-card/20 hover:border-primary/30 transition-all group">
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary group-hover:scale-110 transition-transform">
                            <Plus size={16} />
                          </div>
                          <p className="font-mono text-sm text-muted-foreground group-hover:text-primary">DEPLOY NEW AGENT CLUSTER</p>
                        </div>
                      </Card>
                    </div>
                  </div>
                </TabsContent>

                {/* NEW AGENT FOUNDRY MODULE */}
                <TabsContent value="foundry" className="mt-0 h-full border-none p-0">
                  <div className="space-y-6">
                    <div>
                      <h2 className="text-2xl font-display font-bold text-purple-400 mb-1">Agent Foundry</h2>
                      <p className="text-muted-foreground">Advanced skill creation, file access control, and tool allocation.</p>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[600px]">
                      
                      {/* Left Column: File & Skill Management */}
                      <div className="space-y-4 lg:col-span-2 flex flex-col h-full">
                        {/* File Access Manager */}
                        <Card className="bg-card/30 border-border/50 backdrop-blur-sm flex-1">
                          <CardHeader className="py-3">
                            <CardTitle className="font-mono text-sm flex items-center gap-2">
                              <Folder size={16} className="text-yellow-500" /> FILE ACCESS SCOPE
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="h-[200px]">
                            <ScrollArea className="h-full pr-4">
                              <div className="space-y-2">
                                {["src/components", "src/lib", "public/assets", "server/routes"].map((path) => (
                                  <div key={path} className="flex items-center justify-between p-2 rounded bg-black/20 border border-white/5 hover:border-primary/30 transition-colors group">
                                    <div className="flex items-center gap-2 text-xs font-mono">
                                      <Folder size={12} className="text-muted-foreground group-hover:text-primary" />
                                      {path}
                                    </div>
                                    <div className="flex gap-2">
                                      <Badge variant="outline" className="text-[10px] h-5 bg-green-500/10 text-green-400 border-green-500/20">READ</Badge>
                                      <Badge variant="outline" className="text-[10px] h-5 bg-orange-500/10 text-orange-400 border-orange-500/20">WRITE</Badge>
                                    </div>
                                  </div>
                                ))}
                                <Button variant="ghost" size="sm" className="w-full text-xs text-muted-foreground border-dashed border border-border mt-2 h-8">
                                  + ADD PATH
                                </Button>
                              </div>
                            </ScrollArea>
                          </CardContent>
                        </Card>

                        {/* Skill Builder */}
                        <Card className="bg-card/30 border-border/50 backdrop-blur-sm flex-[1.5]">
                          <CardHeader className="py-3">
                            <CardTitle className="font-mono text-sm flex items-center gap-2">
                              <Sparkles size={16} className="text-purple-500" /> SKILL FABRICATOR
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                              <div className="space-y-2">
                                <Label className="text-xs">SKILL NAME</Label>
                                <Input placeholder="e.g. DataScraper" className="h-8 bg-black/20 font-mono text-xs" />
                              </div>
                              <div className="space-y-2">
                                <Label className="text-xs">TRIGGER TYPE</Label>
                                <div className="flex gap-2">
                                  <Badge variant="secondary" className="cursor-pointer hover:bg-primary/20">ON_COMMAND</Badge>
                                  <Badge variant="outline" className="cursor-pointer hover:bg-primary/20">CRON</Badge>
                                  <Badge variant="outline" className="cursor-pointer hover:bg-primary/20">EVENT</Badge>
                                </div>
                              </div>
                            </div>
                            <div className="space-y-2">
                              <Label className="text-xs">LOGIC DEFINITION (PYTHON/JS)</Label>
                              <div className="h-32 bg-black/40 rounded border border-white/10 p-2 font-mono text-xs text-muted-foreground">
                                // Define skill logic here...
                                <br />
                                <span className="text-purple-400">def</span> <span className="text-yellow-400">execute</span>(context):
                                <br />
                                &nbsp;&nbsp;<span className="text-purple-400">return</span> context.data
                              </div>
                            </div>
                            <div className="flex justify-end gap-2">
                              <Button size="sm" variant="outline" className="h-7 text-xs">TEST</Button>
                              <Button size="sm" className="h-7 text-xs bg-purple-600 hover:bg-purple-700">COMPILE SKILL</Button>
                            </div>
                          </CardContent>
                        </Card>
                      </div>

                      {/* Right Column: Tool Allocation */}
                      <Card className="bg-card/30 border-border/50 backdrop-blur-sm flex flex-col h-full">
                        <CardHeader className="py-3">
                          <CardTitle className="font-mono text-sm flex items-center gap-2">
                            <Hammer size={16} className="text-blue-500" /> TOOLKIT ALLOCATION
                          </CardTitle>
                          <CardDescription className="text-xs">Drag tools to assign to active agent</CardDescription>
                        </CardHeader>
                        <CardContent className="flex-1 overflow-y-auto">
                          <div className="space-y-4">
                            <div className="p-3 bg-black/20 rounded border border-primary/20 mb-4">
                              <div className="text-[10px] text-primary uppercase tracking-wider mb-1">TARGET AGENT</div>
                              <div className="font-bold text-sm">CODER-ALPHA</div>
                            </div>

                            <div className="space-y-2">
                              {tools.map((tool) => (
                                <div key={tool.id} className="flex items-center justify-between p-3 rounded bg-card/50 border border-border hover:border-primary/50 cursor-grab active:cursor-grabbing transition-all group">
                                  <div className="flex items-center gap-3">
                                    <div className="p-2 rounded bg-black/40 text-muted-foreground group-hover:text-primary transition-colors">
                                      {tool.icon}
                                    </div>
                                    <div>
                                      <div className="font-bold text-xs">{tool.name}</div>
                                      <div className="text-[10px] text-muted-foreground">{tool.category}</div>
                                    </div>
                                  </div>
                                  <Switch />
                                </div>
                              ))}
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="customization" className="mt-0 h-full border-none p-0">
                  <div className="space-y-6">
                    <div>
                      <h2 className="text-2xl font-display font-bold text-pink-500 mb-1">Customizations</h2>
                      <p className="text-muted-foreground">Personalize your terminal experience.</p>
                    </div>

                    <Card className="bg-card/30 border-border/50 backdrop-blur-sm">
                      <CardHeader>
                        <CardTitle className="font-mono text-lg">Theme Settings</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-3 gap-4">
                          {[
                            { name: "Cyberpunk", color: "bg-cyan-500" },
                            { name: "Matrix", color: "bg-green-500" },
                            { name: "Sunset", color: "bg-orange-500" }
                          ].map((theme) => (
                            <div key={theme.name} className="border border-border rounded-md p-3 cursor-pointer hover:border-primary/50 transition-colors bg-black/20">
                              <div className={`w-full h-20 rounded-md mb-2 ${theme.color} opacity-20`} />
                              <div className="font-mono text-xs text-center font-bold">{theme.name}</div>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-card/30 border-border/50 backdrop-blur-sm">
                      <CardHeader>
                        <CardTitle className="font-mono text-lg">Interface Options</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <Label>CRT Scanlines</Label>
                            <p className="text-xs text-muted-foreground">Simulate retro monitor effects</p>
                          </div>
                          <Switch defaultChecked />
                        </div>
                        <Separator className="bg-border/30" />
                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <Label>Sound Effects</Label>
                            <p className="text-xs text-muted-foreground">UI interaction sounds</p>
                          </div>
                          <Switch />
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>
              </motion.div>
            </AnimatePresence>
          </div>
        </Tabs>
      </div>
    </div>
  );
}

