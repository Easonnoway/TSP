    cd /TSP/RQ_Data/rq2/inference

    python ./inference_with_template.py \
        --model model/CodeLlama-7b-Instruct-hf \
        --input TSP/RQ_Data/rq2/dataset.json \
        --output TSP/RQ_Data/rq2/inference/output/codellama7b_inference_output.json

    # Convert to database 
    python /TSP/RQ_Data/rq2/inference/convert_to_database.py
