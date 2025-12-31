import { motion } from "framer-motion";
import { User, Settings, Users, Palette, Shield, ChevronRight } from "lucide-react";
import { useLocation } from "wouter";

export default function Admin() {
  const [_, setLocation] = useLocation();

  const menuItems = [
    { 
      id: "profile", 
      title: "User Profile", 
      icon: <User className="text-primary" size={24} />,
      description: "Manage personal settings and credentials"
    },
    { 
      id: "project", 
      title: "Project Settings", 
      icon: <Settings className="text-secondary" size={24} />,
      description: "Configure environment variables and build pipelines"
    },
    { 
      id: "agents", 
      title: "Agent Management", 
      icon: <Users className="text-green-400" size={24} />,
      description: "Orchestrate active agents and resource allocation"
    },
    { 
      id: "customization", 
      title: "Customizations", 
      icon: <Palette className="text-pink-500" size={24} />,
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
      <div className="flex-1 overflow-y-auto p-8 relative z-10 max-w-5xl mx-auto w-full">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8"
        >
          <h2 className="text-2xl font-display font-bold mb-2">System Configuration</h2>
          <p className="text-muted-foreground">Select a module to configure system parameters and access controls.</p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {menuItems.map((item, index) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="group relative overflow-hidden p-6 rounded-lg border border-border bg-card/10 hover:bg-card/30 hover:border-primary/50 transition-all duration-300 cursor-pointer"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-primary/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
              
              <div className="flex items-start justify-between relative z-10">
                <div className="flex items-start gap-4">
                  <div className="p-3 rounded-md bg-black/40 border border-white/5 group-hover:scale-110 transition-transform duration-300">
                    {item.icon}
                  </div>
                  <div>
                    <h3 className="font-bold text-lg mb-1 group-hover:text-primary transition-colors">{item.title}</h3>
                    <p className="text-sm text-muted-foreground">{item.description}</p>
                  </div>
                </div>
                <ChevronRight className="text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all" />
              </div>

              {/* Decorative corners */}
              <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-primary/0 group-hover:border-primary/50 transition-colors" />
              <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-primary/0 group-hover:border-primary/50 transition-colors" />
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
