import { motion } from "framer-motion";
import matrixSphere from "@assets/generated_images/matrix_code_sphere.png";

export function SphereSpinner({ isActive }: { isActive: boolean }) {
  return (
    <div className="relative w-full aspect-square flex items-center justify-center">
      <motion.div
        className="relative w-[80%] h-[80%] rounded-full overflow-hidden shadow-[0_0_30px_rgba(34,197,94,0.3)]"
        animate={isActive ? { rotate: 360 } : { rotate: 0 }}
        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
      >
        <div className="absolute inset-0 bg-black/60 z-10" />
        <img 
          src={matrixSphere} 
          alt="Neural Core" 
          className="absolute inset-0 w-full h-full object-cover mix-blend-screen opacity-80"
        />
        
        {/* Glowing Core */}
        <div className="absolute inset-0 bg-radial-gradient from-green-500/20 via-transparent to-transparent z-20" />
      </motion.div>

      {/* Orbital Rings */}
      <motion.div 
        className="absolute w-full h-full border border-green-500/30 rounded-full"
        animate={{ rotateX: 60, rotateZ: 360 }}
        transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
      />
      <motion.div 
        className="absolute w-[110%] h-[110%] border border-green-500/20 rounded-full border-dashed"
        animate={{ rotateY: 60, rotateZ: -360 }}
        transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
      />
    </div>
  );
}
