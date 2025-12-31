import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

export function MatrixLoader({ onComplete }: { onComplete: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [progress, setProgress] = useState(0);

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
      ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = "#0F0"; // Green text
      ctx.font = `${fontSize}px monospace`;

      for (let i = 0; i < drops.length; i++) {
        const text = chars[Math.floor(Math.random() * chars.length)];
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);

        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }
        drops[i]++;
      }
      
      frameId = requestAnimationFrame(draw);
    };

    draw();

    // Progress simulation
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(onComplete, 500);
          return 100;
        }
        return prev + 1;
      });
    }, 30);

    return () => {
      cancelAnimationFrame(frameId);
      clearInterval(interval);
    };
  }, [onComplete]);

  return (
    <div className="fixed inset-0 z-[100] bg-black text-green-500 font-mono flex flex-col items-center justify-center">
      <canvas ref={canvasRef} className="absolute inset-0 opacity-40" />
      
      <div className="relative z-10 text-center space-y-4">
        <motion.div 
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="text-4xl font-bold tracking-[0.5em] text-shadow-glow"
        >
          NEXUS
        </motion.div>
        
        <div className="w-64 h-1 bg-green-900 rounded-full overflow-hidden">
          <motion.div 
            className="h-full bg-green-500 shadow-[0_0_10px_#22c55e]"
            style={{ width: `${progress}%` }}
          />
        </div>
        
        <div className="text-xs tracking-widest opacity-70">
          INITIALIZING NEURAL INTERFACE... {progress}%
        </div>
      </div>
    </div>
  );
}
