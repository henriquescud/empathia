# **👁️ EmpathIA: Sistema de Análise e Reconhecimento Facial de Emoções**
## Este projeto foi desenvolvido para demonstrar a aplicação de Visão Computacional e Deep Learning (via DeepFace) na identificação de funcionários e na análise em tempo real das emoções expressas. 
Os dados de cadastro e embeddings faciais são armazenados de forma segura no **SQLite3**.

### 🚀 Funcionalidades Principais
**- Cadastro de Funcionário**: Captura facial via webcam e registro dos dados do funcionário (**Nome, Cargo, Setor**) junto ao seu vetor facial (embedding) no **SQLite3**.

**- Reconhecimento Facial**: Identificação em tempo real do funcionário cadastrado no banco de dados.

**- Análise de Emoção**: Detecção e exibição em tempo real das oito emoções primárias (alegria, tristeza, raiva, surpresa, etc.).

**- Ambiente de Produção**: Uso de **SQLite3** (nuvem) para persistência e **PyQT5** para a interface gráfica.
