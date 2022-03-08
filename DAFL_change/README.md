
## One generator for all
- Download model, putit under "cache/models/"
  - [GoogleDrive](https://drive.google.com/file/d/1J1JDCnF3mp23sNUqESLxejf1XaVXS8Kj/view?usp=sharing)
- small batch
  ```bash
  run_small.sh
  ```
- large batch
  ```bash
  run_large.sh
  ```

## N generators for N classes

- Download model, rename folder to "cache/models"
  - [GoogleDrive](https://drive.google.com/drive/folders/1L94fyrLCpbGFLQi5Q94_1MdYJp6Tb0kI?usp=sharing)
    ```
    ├─ DAFLDeepinvert-train_v6_tmp.py
    │
    └─ cache/
        │
        ├─ ckpts/
        │   │
        │   └── multi/
        │  
        ├─ data/
        │
        │
        └── models/
    ``` 