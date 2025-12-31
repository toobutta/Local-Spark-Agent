import { motion } from "framer-motion";

export function LegoLoader() {
  // Create a 5x5 grid of blocks
  const blocks = Array.from({ length: 25 });

  return (
    <div className="flex flex-col items-center justify-center p-4 space-y-2">
      <div className="text-xs font-mono text-primary animate-pulse">BUILDING ARTIFACTS...</div>
      <div className="grid grid-cols-5 gap-1">
        {blocks.map((_, i) => (
          <motion.div
            key={i}
            className="w-4 h-4 bg-primary/80 shadow-[0_0_5px_rgba(0,255,255,0.5)]"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ 
              scale: [0, 1, 1, 0], 
              opacity: [0, 1, 1, 0] 
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              delay: (i % 5) * 0.1 + Math.floor(i / 5) * 0.1,
              ease: "easeInOut",
              times: [0, 0.2, 0.8, 1]
            }}
          />
        ))}
      </div>
    </div>
  );
}
