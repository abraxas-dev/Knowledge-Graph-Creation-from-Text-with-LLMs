from src.pipeline import Pipeline

def main():
    config_path = "./config.json"
    pipeline = Pipeline(config_path)
    pipeline.run()

if __name__ == "__main__":
    main()
