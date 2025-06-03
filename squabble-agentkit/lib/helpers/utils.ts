import path from "node:path";
import fs from "node:fs";

export function validateEnvironment(vars: string[]): Record<string, string> {
    const missing = vars.filter((v) => !process.env[v]);
  
    if (missing.length) {
      try {
        const envPath = path.resolve(process.cwd(), ".env");
        if (fs.existsSync(envPath)) {
          const envVars = fs
            .readFileSync(envPath, "utf-8")
            .split("\n")
            .filter((line) => line.trim() && !line.startsWith("#"))
            .reduce<Record<string, string>>((acc, line) => {
              const [key, ...val] = line.split("=");
              if (key && val.length) acc[key.trim()] = val.join("=").trim();
              return acc;
            }, {});
  
          missing.forEach((v) => {
            if (envVars[v]) process.env[v] = envVars[v];
          });
        }
      } catch (e) {
        console.error(e);
        /* ignore errors */
      }
  
      const stillMissing = vars.filter((v) => !process.env[v]);
      if (stillMissing.length) {
        console.error("Missing env vars:", stillMissing.join(", "));
        process.exit(1);
      }
    }
  
    return vars.reduce<Record<string, string>>((acc, key) => {
      acc[key] = process.env[key] as string;
      return acc;
    }, {});
  }