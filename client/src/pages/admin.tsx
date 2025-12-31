import { motion, AnimatePresence } from "framer-motion";
import { User, Settings, Users, Palette, Shield, ChevronRight, LayoutGrid } from "lucide-react";
import { useLocation } from "wouter";
import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";

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
      id: "project", 
      title: "Project Settings", 
      icon: <Settings size={18} />,
      description: "Configure environment variables and build pipelines"
    },
    { 
      id: "agents", 
      title: "Agent Management", 
      icon: <Users size={18} />,
      description: "Orchestrate active agents and resource allocation"
    },
    { 
      id: "customization", 
      title: "Customizations", 
      icon: <Palette size={18} />,
      description: "Theme editor and UI preferences"
    }
  ];

  return (
    <div className="min-h-screen bg-background text-foreground font-mono overflow-hidden flex flex-col relative">
      <div className="scanline" />
      <div className="pointer-events-none fixed inset-0 z-50 bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.4)_100%)]" />

      {/* Header */}
      <header className="h-16 border-b border-border/50 bg-card/20 flex items-center justify-between px-6 backdrop-blur-md relative z-10">
        <div className="flex items-center gap-3 text-primary cursor-pointer hover:opacity-80 transition-opacity" onClick={() => setLocation("/")}>
          <Shield size={20} />
          <h1 className="font-display font-bold tracking-widest text-xl">NEXUS ADMIN</h1>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground bg-black/40 px-3 py-1.5 rounded border border-white/5">
          <div className="w-2 h-2 rounded-full bg-orange-500 animate-pulse" />
          ADMIN MODE
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

                <TabsContent value="project" className="mt-0 h-full border-none p-0">
                  <div className="space-y-6">
                    <div>
                      <h2 className="text-2xl font-display font-bold text-secondary mb-1">Project Settings</h2>
                      <p className="text-muted-foreground">Configure global variables and build environments.</p>
                    </div>

                    <Card className="bg-card/30 border-border/50 backdrop-blur-sm">
                      <CardHeader>
                        <CardTitle className="font-mono text-lg">Environment Variables</CardTitle>
                        <CardDescription>Manage keys and secrets for the Nexus runtime.</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="space-y-2">
                          <Label>API_KEY_OPENAI</Label>
                          <div className="flex gap-2">
                            <Input type="password" value="sk-........................" readOnly className="bg-black/20 border-border/50 font-mono flex-1" />
                            <Button variant="outline" className="border-primary/20 hover:bg-primary/10 text-primary">Reveal</Button>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <Label>DGX_CLUSTER_ID</Label>
                          <Input defaultValue="DGX-A100-SPARK-09" className="bg-black/20 border-border/50 font-mono" />
                        </div>
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
                </TabsContent>

                <TabsContent value="agents" className="mt-0 h-full border-none p-0">
                  <div className="space-y-6">
                    <div>
                      <h2 className="text-2xl font-display font-bold text-green-400 mb-1">Agent Management</h2>
                      <p className="text-muted-foreground">Monitor and orchestrate your autonomous workforce.</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {["CODER-ALPHA", "SECURITY-PRIME", "DATA-SENTRY"].map((agent) => (
                        <Card key={agent} className="bg-card/30 border-border/50 backdrop-blur-sm overflow-hidden relative">
                          <div className="absolute top-0 right-0 p-2">
                            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                          </div>
                          <CardHeader className="pb-2">
                            <CardTitle className="font-mono text-md">{agent}</CardTitle>
                            <CardDescription className="text-xs">Uptime: 42h 12m</CardDescription>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-2 text-xs font-mono">
                              <div className="flex justify-between">
                                <span>Task Queue</span>
                                <span className="text-primary">Idle</span>
                              </div>
                              <div className="w-full bg-black/40 h-1.5 rounded-full overflow-hidden">
                                <div className="bg-green-500/50 h-full w-[20%]" />
                              </div>
                            </div>
                            <div className="mt-4 flex gap-2">
                              <Button size="sm" variant="outline" className="h-7 text-xs border-destructive/50 text-destructive hover:bg-destructive/10">TERMINATE</Button>
                              <Button size="sm" variant="outline" className="h-7 text-xs border-primary/50 text-primary hover:bg-primary/10">LOGS</Button>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                      
                      <Card className="bg-card/10 border-dashed border-border flex items-center justify-center min-h-[160px] cursor-pointer hover:bg-card/20 hover:border-primary/30 transition-all group">
                        <div className="text-center space-y-2">
                          <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center mx-auto text-primary group-hover:scale-110 transition-transform">
                            <Users size={20} />
                          </div>
                          <p className="font-mono text-sm text-muted-foreground group-hover:text-primary">DEPLOY NEW AGENT</p>
                        </div>
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

