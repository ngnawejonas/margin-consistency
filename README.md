# Margin Consistency
Paper: "Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers."

1/`$pip install -r requirements.txt`

2/edit the yaml file:
   * specify the attack (fab for fab attack, cw for carlini-wagner, clever for clever score or otherwise for auto-attack) 
   * a folder to save the results
  
3/`$python3 eval.py`

Or use `run.sh` file