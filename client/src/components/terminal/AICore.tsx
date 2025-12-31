import { motion } from "framer-motion";
import aiCoreImg from "@assets/generated_images/cyberpunk_ai_core_visualization.png";

export function AICore() {
  return (
    <div className="relative w-full aspect-square overflow-hidden rounded-lg border border-primary/30 bg-black/50">
      {/* Background Image */}
      <img 
        src={aiCoreImg} 
        alt="AI Core" 
        className="absolute inset-0 w-full h-full object-cover opacity-60 mix-blend-screen"
      />
      
      {/* Animated Overlay Rings */}
      <div className="absolute inset-0 flex items-center justify-center">
        <motion.div 
          className="w-[80%] h-[80%] rounded-full border border-primary/40 border-dashed"
          animate={{ rotate: 360 }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
        />
        <motion.div 
          className="w-[60%] h-[60%] rounded-full border border-secondary/40 border-dotted"
          animate={{ rotate: -360 }}
          transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
        />
      </div>

      {/* Status Text Overlay */}
      <div className="absolute bottom-2 left-2 right-2 flex justify-between text-[10px] font-mono text-primary/80">
        <span>CORE: ONLINE</span>
        <span className="animate-pulse">PROCESSING</span>
      </div>
      
      {/* Scanline overlay for this specific component */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,20,0)50%,rgba(0,0,0,0.25)50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_2px,3px_100%] pointer-events-none opacity-50"></div>
    </div>
  );
}
