import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { Folder, ChevronRight, Terminal } from "lucide-react";

export function MatrixLoader({ 
  onComplete, 
  onProjectSelect 
}: { 
  onComplete: () => void;
  onProjectSelect?: (projectName: string) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [progress, setProgress] = useState(0);
  const [showProjects, setShowProjects] = useState(false);

  const recentProjects = [
    { id: "genesis", name: "Project Genesis", path: "~/workspace/genesis" },
    { id: "dgx-spark", name: "SparkPlug DGX", path: "~/compute/dgx-spark" },
    { id: "neural-net", name: "Neural Interface v2", path: "~/research/neural" },
  ];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%^&*";
    const fontSize = 14;
    const columns = canvas.width / fontSize;
    const drops: number[] = [];

    for (let i = 0; i < columns; i++) {
      drops[i] = 1;
    }

    let frameId: number;

    const draw = () => {
      ctx.fillStyle = "rgba(0, 0, 0, 0.1)"; // Increased fade for slower trail
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = "#0F0"; // Green text
      ctx.font = `${fontSize}px monospace`;

      for (let i = 0; i < drops.length; i++) {
        const text = chars[Math.floor(Math.random() * chars.length)];
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);

        if (drops[i] * fontSize > canvas.height && Math.random() > 0.99) { // Reduced drop density
          drops[i] = 0;
        }
        
        // Slower movement - only update every other frame or use a fractional increment
        if (Math.random() > 0.5) { 
          drops[i]++;
        }
      }
      
      frameId = requestAnimationFrame(draw);
    };

    draw();

    // Progress simulation
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(() => setShowProjects(true), 500);
          return 100;
        }
        return prev + 1;
      });
    }, 20);

    return () => {
      cancelAnimationFrame(frameId);
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="fixed inset-0 z-[100] bg-black text-green-500 font-mono flex flex-col items-center justify-center">
      <canvas ref={canvasRef} className="absolute inset-0 opacity-40" />
      
      <AnimatePresence mode="wait">
        {!showProjects ? (
          <motion.div 
            key="loader"
            initial={{ opacity: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="relative z-10 flex flex-col items-center space-y-4"
          >
            <motion.div 
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5 }}
              className="text-4xl font-bold tracking-[0.5em] text-shadow-glow"
            >
              SPARKPLUG
            </motion.div>
            
            <div className="w-full h-1 bg-green-900 rounded-full overflow-hidden">
              <motion.div 
                className="h-full bg-green-500 shadow-[0_0_10px_#22c55e]"
                style={{ width: `${progress}%` }}
              />
            </div>
            
            <div className="text-xs tracking-widest opacity-70 animate-pulse">
              INITIALIZING NEURAL INTERFACE... {progress}%
            </div>
          </motion.div>
        ) : (
          <motion.div 
            key="projects"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="relative z-10 w-full max-w-md p-6 bg-black/80 border border-green-500/30 backdrop-blur-xl rounded-lg shadow-[0_0_30px_rgba(0,255,0,0.1)]"
          >
            <div className="flex items-center gap-2 mb-6 border-b border-green-500/20 pb-4">
              <Terminal size={20} className="text-green-500" />
              <h2 className="text-sm font-bold tracking-widest uppercase">Select Workspace</h2>
            </div>

            <div className="space-y-3">
              <div className="text-[10px] text-green-500/50 uppercase tracking-tighter mb-1">Recent Projects</div>
              {recentProjects.map((project) => (
                <button
                  key={project.id}
                  onClick={() => onProjectSelect?.(project.name)}
                  className="w-full group flex items-center justify-between p-3 bg-green-500/5 border border-green-500/10 rounded hover:bg-green-500/20 hover:border-green-500/40 transition-all duration-300"
                >
                  <div className="flex items-center gap-3">
                    <Folder size={16} className="text-green-500/70 group-hover:text-green-500" />
                    <div className="text-left">
                      <div className="text-xs font-bold text-green-400 group-hover:text-green-300">{project.name}</div>
                      <div className="text-[10px] text-green-500/40 group-hover:text-green-500/60 font-mono">{project.path}</div>
                    </div>
                  </div>
                  <ChevronRight size={14} className="text-green-500/30 group-hover:translate-x-1 transition-transform" />
                </button>
              ))}

              <button
                onClick={onComplete}
                className="w-full mt-4 flex items-center justify-center gap-2 py-3 border border-dashed border-green-500/30 rounded text-[10px] text-green-500/60 hover:text-green-400 hover:bg-green-500/5 hover:border-green-500/50 transition-all"
              >
                SKIP TO CHAT INTERFACE <ChevronRight size={12} />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
