import 'dotenv/config';

const DEFAULT_TEMPERATURE = 0.2;
const DEFAULT_MAX_RETRIES = 2;

const config = {
  get baseUrl(): string {
    return process.env.BASE_URL ?? 'http://localhost:11434';
  },

  get modelName(): string {
    return process.env.MODEL_NAME ?? 'llama3.1';
  },

  get multimodalModelName(): string {
    return process.env.MULTIMODAL_MODEL_NAME ?? 'llava:13b';
  },

  get temperature(): number {
    if (!process.env.TEMPERATURE) return DEFAULT_TEMPERATURE;

    const temperature = parseFloat(process.env.TEMPERATURE);
    if (isNaN(temperature) || temperature < 0 || temperature > 1) {
      return DEFAULT_TEMPERATURE;
    }

    return temperature;
  },

  get maxRetries(): number {
    if (!process.env.MAX_RETRIES) return DEFAULT_MAX_RETRIES;

    const maxRetries = parseInt(process.env.MAX_RETRIES, 10);
    if (isNaN(maxRetries) || maxRetries < 0) {
      return DEFAULT_MAX_RETRIES;
    }

    return maxRetries;
  },
};

export default config;
