from data_cleaner_module import build_pipeline, save_pipeline

if __name__ == '__main__':
    pipeline = build_pipeline()
    save_pipeline(pipeline)
    print("✅ Data cleaning pipeline saved.")