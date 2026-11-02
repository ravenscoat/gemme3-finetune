"""Previously completed Bitext results, read without recomputing them."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"outputs/bitext-4b-full-pass-evaluation-500-windows"
def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def metrics():
    pilot = read_json(ROOT/'outputs/bitext-4b-evaluation-500-windows/evaluation_unfenced.json')
    full = read_json(RUN/'evaluation_unfenced.json')
    pm = read_json(ROOT/'outputs/bitext-4b-evaluation-500-windows/mauve_evaluation.json')
    fm = read_json(RUN/'mauve_evaluation.json')
    import statistics
    values = []
    for title, evaluation, mauve, label in [('Original',pilot,pm,'base'), ('Pilot',pilot,pm,'tuned'), ('Full pass',full,fm,'tuned')]:
        e = evaluation[label]
        values.append({'name':title, 'valid':e['valid'], 'attempted':e['attempted'],
            'intents':len(e['intent_counts']), 'intent_tv':e['distribution_distances']['intent_total_variation'],
            'category_tv':e['distribution_distances']['category_total_variation'],
            'request_copy':e['training_request_match_rate'], 'reply_copy':e['training_response_match_rate'],
            'record_copy':e['exact_training_record_match_rate'],
            'request_mauve':statistics.median(x['mauve'] for x in mauve['fields']['instruction']['scores'][label]),
            'reply_mauve':statistics.median(x['mauve'] for x in mauve['fields']['response']['scores'][label])})
    return {'stages':values, 'test':read_json(ROOT/'outputs/final-assessment/final_test_results.json'),
            'intent_counts':full['tuned']['intent_counts'], 'train_per_intent':800,
            'ai_review':{'sample':20, 'pass':19, 'unsure':1, 'expert':False}}
