# SLM_Manager Quick Reference

## Build the model

** To modify the behavior of the SLM, you can change the content of the Modelfile. Make sure however to run the following command when you're done. **

```bash
// From root:
ollama create bira-assistant -f Modelfile // Necessary because we don't use system Prompt anymore. Everytime you modify Modelfile, please run this command again.
```
More informations on the modelfile can be found here: https://docs.ollama.com/modelfile


## Run BIRA

```bash
// From src:
python main.py --mock --SLM_DEBUG // SLM_DEBUG to see details from how the SLM ats
```

## Optional checks

```bash
ollama list | grep bira
ollama run bira-assistant "give me a bottle"
```
