## 14/08

Podem ficar contenedors per cada runtime pq ara sol detecta llama.cpp no podem executar amb Transformers (PyTorch)
                                                                     JAULL CORE
                                                                        │
                                                                Runtime Manager
                                                                        │
                                                                      ExecutionPlan
                                                                        │
                                                    ┌──────────────────┴──────────────────┐
                                                    │                                     │
                                            LlamaCppAdapter                     TransformersAdapter
                                                    │                                     │
                                                ┌─────┴─────┐                         ┌─────┴─────┐
                                                │           │                         │           │
                                            Native      Container                  Native      Container
                                                │           │                         │           │
                                            llama-cli   llama.cpp                PyTorch      PyTorch
                                                        container                             container