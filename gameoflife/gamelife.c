#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N_GLOBAL 40         // dimensão da matriz (40x40), deve ser um num divisivel por 4
#define PROCESSOS_LADO 4    // quantidade de processos por linha/coluna na grade (4x4)
#define NUM_PROCESSOS 16    // total de processos (4 * 4)
#define GHOST 5             // largura da borda de celulas fantasma
#define GERACOES 10         // num de geracoes da simulacao
#define VIVA 1
#define MORTA 0

// mapear coordenadas 2D em um vetor 1D
#define IDX(i, j) ((i) * colunas_total + (j))

void imprimir_mapa_processos(){
    printf("\n- DIVISAO DOS PROCESSOS -\n\n");
    printf("+----+----+----+----+\n");
    printf("| P0 | P1 | P2 | P3 |\n");
    printf("+----+----+----+----+\n");
    printf("| P4 | P5 | P6 | P7 |\n");
    printf("+----+----+----+----+\n");
    printf("| P8 | P9 |P10 |P11 |\n");
    printf("+----+----+----+----+\n");
    printf("|P12 |P13 |P14 |P15 |\n");
    printf("+----+----+----+----+\n");
}

// imprime a matriz global completa, executada apenas pelo processo 0
void imprimir_matriz_global(int *matriz_global, int geracao, int populacao){
    printf("\n- GERACAO %d | POPULACAO TOTAL: %d -\n\n", geracao, populacao);

    for (int i = 0; i < N_GLOBAL; i++){
        for (int j = 0; j < N_GLOBAL; j++)
            // imprime O para células vivas e . para mortas
            printf("%c", matriz_global[i * N_GLOBAL + j] ? 'O' : '.');
        printf("\n");
    }
}


// imprime a matriz local de um processo específico, evidenciando as Ghost Cells.
void imprimir_processo_com_ghosts(int *matriz, int linhas_total, int colunas_total,
                                  int rank, int proc_linha, int proc_coluna){
    printf("\n- PROCESSO %d | posicao (%d,%d) -\n", rank, proc_linha, proc_coluna);
    printf("G = ghost cell | O = viva | . = morta\n\n");

    for (int i = 0; i < linhas_total; i++){
        for (int j = 0; j < colunas_total; j++){
            // Define se a posição atual pertence a area de Ghost Cells
            int eh_ghost = i < GHOST || i >= linhas_total - GHOST ||
                           j < GHOST || j >= colunas_total - GHOST;

            if (eh_ghost)
                printf("G");
            else
                printf("%c", matriz[i * colunas_total + j] ? 'O' : '.');
        }
        printf("\n");
    }
}

// ============================================================================
// LÓGICA INTERNA DO JOGO DA VIDA
// ============================================================================
// ignora as Ghost Cells, e conta quantas celulas estao vivas
int contar_populacao_local(int *matriz, int n_local, int colunas_total){
    int total = 0;

    // loop começa em GHOST e vai ate GHOST + n_local para evitar as bordas
    for (int i = GHOST; i < GHOST + n_local; i++)
        for (int j = GHOST; j < GHOST + n_local; j++)
            total += matriz[i * colunas_total + j];

    return total;
}


// recolhe os pedaços de matrizes locais de todos os processos utilizando MPI_Gather e
// reconstroi a matriz original de tamanho N_GLOBAL x N_GLOBAL e chama a impressao
void juntar_e_imprimir_global(int *matriz, int n_local, int colunas_total,
                              int rank, MPI_Comm cart_comm,
                              int geracao, int populacao_total){
    
    // memoria temporaria para pegar a parte real
    int *bloco_local = malloc(n_local * n_local * sizeof(int));

    for (int i = 0; i < n_local; i++)
        for (int j = 0; j < n_local; j++)
            bloco_local[i * n_local + j] = matriz[(i + GHOST) * colunas_total + (j + GHOST)];

    int *todos_blocos = NULL;

    // so o processo 0 precisa de espaço para receber todos os dados
    if (rank == 0)
        todos_blocos = malloc(N_GLOBAL * N_GLOBAL * sizeof(int));

    // Concentra as regioes reais de todos os processos no Processo 0
    MPI_Gather(
        bloco_local, n_local * n_local, MPI_INT, // Envio (todos)
        todos_blocos, n_local * n_local, MPI_INT,// Recebimento (apenas 0)
        0, cart_comm
    );

    // O Processo 0 remonta o quebra-cabeca usando as coordenadas da grade cartesiana
    if (rank == 0){
        int matriz_global[N_GLOBAL * N_GLOBAL];

        for (int r = 0; r < NUM_PROCESSOS; r++){
            int coord[2];
            MPI_Cart_coords(cart_comm, r, 2, coord); // descobre onde o processo r ta na grade

            int inicio_linha = coord[0] * n_local;
            int inicio_coluna = coord[1] * n_local;

            // coloca o bloco do processo r no lugar certo da matriz
            for (int i = 0; i < n_local; i++){
                for (int j = 0; j < n_local; j++){
                    matriz_global[(inicio_linha + i) * N_GLOBAL + inicio_coluna + j] =
                        todos_blocos[r * n_local * n_local + i * n_local + j];
                }
            }
        }

        imprimir_matriz_global(matriz_global, geracao, populacao_total);
        free(todos_blocos);
    }

    free(bloco_local);
}

// computa as regras classicas do jogo de Conway para a próxima geracao
// a leitura eh feita na matriz (que já possui as ghosts atualizadas) e escrita na proxima
void calcular_proxima_geracao(int *matriz, int *proxima, int n_local, int colunas_total){
    
    // faz as regras estritamente na zona interna real
    for (int i = GHOST; i < GHOST + n_local; i++){
        for (int j = GHOST; j < GHOST + n_local; j++){
            int vizinhos = 0;

            // percorre os 8 vizinhos ao redor da celula atual
            for (int dl = -1; dl <= 1; dl++){
                for (int dc = -1; dc <= 1; dc++){
                    if (dl == 0 && dc == 0) continue; // pula a celula do meio

                    vizinhos += matriz[(i + dl) * colunas_total + (j + dc)];
                }
            }

            // aplica as regras do jogo
            if (matriz[i * colunas_total + j] == VIVA)
                proxima[i * colunas_total + j] = (vizinhos == 2 || vizinhos == 3) ? VIVA : MORTA;
            else
                proxima[i * colunas_total + j] = (vizinhos == 3) ? VIVA : MORTA;
        }
    }

    // atualiza a matriz original copiando os dados calculados na proxima matriz
    for (int i = GHOST; i < GHOST + n_local; i++)
        for (int j = GHOST; j < GHOST + n_local; j++)
            matriz[i * colunas_total + j] = proxima[i * colunas_total + j];
}

// ============================================================================
// COMUNICAÇÃO MPI: TROCA DE BORDAS E CANTOS
// ============================================================================
// sincroniza as Ghost Cells enviando os dados reais das bordas para os vizinhos
// correspondentes e recebendo os deles nas posições fantasma
void trocar_bordas_e_cantos(int *matriz, int n_local, int colunas_total, MPI_Comm cart_comm, 
                            int cima, int baixo, int esquerda, int direita, 
                            int cima_esquerda, int cima_direita, int baixo_esquerda, int baixo_direita){
    
    MPI_Datatype tipo_linhas;
    MPI_Datatype tipo_colunas;
    MPI_Datatype tipo_canto;

    // define os tipos de dados customizados do MPI para enviar blocos de memória nao-continuos
    // MPI_Type_vector(quantidade_blocos, tamanho_bloco, passo_stride, tipo, ponteiro)
    MPI_Type_vector(GHOST, n_local, colunas_total, MPI_INT, &tipo_linhas);
    MPI_Type_commit(&tipo_linhas);

    MPI_Type_vector(n_local, GHOST, colunas_total, MPI_INT, &tipo_colunas);
    MPI_Type_commit(&tipo_colunas);

    MPI_Type_vector(GHOST, GHOST, colunas_total, MPI_INT, &tipo_canto);
    MPI_Type_commit(&tipo_canto);

    // troca de linhas cima / baixo
    // envia borda real superior para cima e recebe na ghost inferior do vizinho de baixo
    MPI_Sendrecv(&matriz[IDX(GHOST, GHOST)], 1, tipo_linhas, cima, 0,
                 &matriz[IDX(GHOST + n_local, GHOST)], 1, tipo_linhas, baixo, 0,
                 cart_comm, MPI_STATUS_IGNORE);

    // envia borda real inferior para baixo e recebe na ghost superior do vizinho de cima
    MPI_Sendrecv(&matriz[IDX(GHOST + n_local - GHOST, GHOST)], 1, tipo_linhas, baixo, 1,
                 &matriz[IDX(0, GHOST)], 1, tipo_linhas, cima, 1,
                 cart_comm, MPI_STATUS_IGNORE);

    // troca de colunas esquerda / direita
    // envia borda real esquerda para a esquerda e recebe na ghost direita do vizinho da direita
    MPI_Sendrecv(&matriz[IDX(GHOST, GHOST)], 1, tipo_colunas, esquerda, 2,
                 &matriz[IDX(GHOST, GHOST + n_local)], 1, tipo_colunas, direita, 2,
                 cart_comm, MPI_STATUS_IGNORE);

    // Envia borda real direita para a direita  e recebe na ghost esquerda do vizinho da esquerda
    MPI_Sendrecv(&matriz[IDX(GHOST, GHOST + n_local - GHOST)], 1, tipo_colunas, direita, 3,
                 &matriz[IDX(GHOST, 0)], 1, tipo_colunas, esquerda, 3,
                 cart_comm, MPI_STATUS_IGNORE);

    // troca de cantos (diagonais)
    // cima-esquerda para baixo-direita
    MPI_Sendrecv(&matriz[IDX(GHOST, GHOST)], 1, tipo_canto, cima_esquerda, 4,
                 &matriz[IDX(GHOST + n_local, GHOST + n_local)], 1, tipo_canto, baixo_direita, 4,
                 cart_comm, MPI_STATUS_IGNORE);

    // cima-direita para baixo-esquerda
    MPI_Sendrecv(&matriz[IDX(GHOST, GHOST + n_local - GHOST)], 1, tipo_canto, cima_direita, 5,
                 &matriz[IDX(GHOST + n_local, 0)], 1, tipo_canto, baixo_esquerda, 5,
                 cart_comm, MPI_STATUS_IGNORE);

    // baixo-esquerda para cima-direita
    MPI_Sendrecv(&matriz[IDX(GHOST + n_local - GHOST, GHOST)], 1, tipo_canto, baixo_esquerda, 6,
                 &matriz[IDX(0, GHOST + n_local)], 1, tipo_canto, cima_direita, 6,
                 cart_comm, MPI_STATUS_IGNORE);

    // baixo-direita para cima-esquerda
    MPI_Sendrecv(&matriz[IDX(GHOST + n_local - GHOST, GHOST + n_local - GHOST)], 1, tipo_canto, baixo_direita, 7,
                 &matriz[IDX(0, 0)], 1, tipo_canto, cima_esquerda, 7,
                 cart_comm, MPI_STATUS_IGNORE);

    // libera memoria
    MPI_Type_free(&tipo_linhas);
    MPI_Type_free(&tipo_colunas);
    MPI_Type_free(&tipo_canto);
}

//numero de argumentos e o vetor de argumentos
int main(int argc, char **argv){
    // inicia o ambiente paralelo
    MPI_Init(&argc, &argv);

    int rank, total_processos;
    // retorna o ranking do processo atual
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // retorna o numero total de processos
    MPI_Comm_size(MPI_COMM_WORLD, &total_processos);

    // valores para a topologia logica
    int dimensoes[2] = {4, 4}; // grid 4x4
    int periodico[2] = {0, 0}; // 0 = sem bordas infinitas

    // define o comunicador para todos os processos
    MPI_Comm cart_comm;
    // cria o comunicador (comunicador_antigo, dimensoes, periodicidade, reorder [nao organizar ranks], novo_comunicador) -> topologia logica
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensoes, periodico, 0, &cart_comm);

    // coordenadas x,y
    int coordenadas[2];
    // retorna as coordenadas do processo atual na grade (comunicador, rank, dim, coordenadas)
    MPI_Cart_coords(cart_comm, rank, 2, coordenadas);
    
    // coordenadas atuais
    int proc_linha = coordenadas[0];
    int proc_coluna = coordenadas[1];

    // identificacao dos vizinhos
    int cima, baixo, esquerda, direita;
    // recebe rank e retorna coordenadas: descobre os vizinhos do processo (comunicador, eixo, deslocamento, vizinho_n, vizinho_p)
    MPI_Cart_shift(cart_comm, 0, 1, &cima, &baixo); //linhas
    MPI_Cart_shift(cart_comm, 1, 1, &esquerda, &direita); //colunas

    // identificacao manual dos vizinhos diagonais
    int cima_esquerda = MPI_PROC_NULL, cima_direita = MPI_PROC_NULL;
    int baixo_esquerda = MPI_PROC_NULL, baixo_direita = MPI_PROC_NULL;

    // coordenada temporaria pro calculo
    int coord_temp[2];

    //se a coordenada ta dentro do grid, retorna o rank

    // diagonal superior esquerda
    coord_temp[0] = proc_linha - 1; coord_temp[1] = proc_coluna - 1;
    if (coord_temp[0] >= 0 && coord_temp[1] >= 0)
        // recebe coords e retorna rank (comunicador, coordenadas, rank)
        MPI_Cart_rank(cart_comm, coord_temp, &cima_esquerda);

    // diagonal superior direita
    coord_temp[0] = proc_linha - 1; coord_temp[1] = proc_coluna + 1;
    if (coord_temp[0] >= 0 && coord_temp[1] < PROCESSOS_LADO)
        MPI_Cart_rank(cart_comm, coord_temp, &cima_direita);

    // diagonal inferior esquerda
    coord_temp[0] = proc_linha + 1; coord_temp[1] = proc_coluna - 1;
    if (coord_temp[0] < PROCESSOS_LADO && coord_temp[1] >= 0)
        MPI_Cart_rank(cart_comm, coord_temp, &baixo_esquerda);

    // diagonal inferior direita
    coord_temp[0] = proc_linha + 1; coord_temp[1] = proc_coluna + 1;
    if (coord_temp[0] < PROCESSOS_LADO && coord_temp[1] < PROCESSOS_LADO)
        MPI_Cart_rank(cart_comm, coord_temp, &baixo_direita);

    // alocacao das matrizes locais (reais + ghost cells)
    // 10x10 pra cada processo
    int n_local = N_GLOBAL / PROCESSOS_LADO;
    // linhas reais + ghost cells de ambos os lados
    int linhas_total   = n_local + 2 * GHOST;
    // colunas reais + ghost cells de ambos os lados
    int colunas_total  = n_local + 2 * GHOST;

    // calloc inicializa o array preenchido com celulas mortas 0
    // matriz atual
    int *matriz  = calloc(linhas_total * colunas_total, sizeof(int));
    // prox. matrix pra calcular a proxima geracao
    int *proxima = calloc(linhas_total * colunas_total, sizeof(int));

    //aprox. o centro pra colocar o padrao inicial
    int deslocamento_linha   = 19;
    int deslocamento_coluna  = 19;
    
    // grid inicial com formato de cruz
    // int padrao[][2] = {
    //             {0, 1},
    //     {1, 0}, {1, 1}, {1, 2},
    //             {2, 1}
    // };

    // padrao que nao muda
    // int padrao[][2] = {
    //     {0,0}, {0,1},
    //     {1,0}, {1,1}
    // };

    // glider
    int padrao[][2] = {
        {0,1},
        {1,2},
        {2,0},
        {2,1},
        {2,2}
    };

    // quantidade de celulas vivas no padrao
    int qtd_padrao = sizeof(padrao) / sizeof(padrao[0]);

    // Cada processo mapeia se o desenho inicial cai dentro do seu territorio
    for (int k = 0; k < qtd_padrao; k++){
        // posicao da celula na matriz real levando em conta o deslocamento
        int linha_global   = padrao[k][0] + deslocamento_linha;
        int coluna_global  = padrao[k][1] + deslocamento_coluna;

        // define quais partes da matriz global cada processo vai cuidar
        int inicio_linha   = proc_linha   * n_local;
        int inicio_coluna  = proc_coluna  * n_local;
        int fim_linha      = inicio_linha  + n_local - 1;
        int fim_coluna     = inicio_coluna + n_local - 1;

        // verifica se a celula viva esta no bloco do processo
        if (linha_global >= inicio_linha && linha_global <= fim_linha &&
            coluna_global >= inicio_coluna && coluna_global <= fim_coluna)
        {
            // converte a posicao global para local (considerando as ghost cells)
            int linha_local   = linha_global  - inicio_linha  + GHOST;
            int coluna_local  = coluna_global - inicio_coluna + GHOST;
            // marca a celula como viva
            matriz[IDX(linha_local, coluna_local)] = VIVA;
        }
    }

    // define um unico processo pra printar as infos
    if (rank == 0){
        printf("Matriz global: %dx%d\n", N_GLOBAL, N_GLOBAL);
        printf("Processos: %d\n", NUM_PROCESSOS);
        printf("Divisao: 4x4 processos\n");
        printf("Cada processo: %dx%d celulas reais\n", n_local, n_local);
        printf("Ghost cells: largura %d ao redor de cada bloco\n", GHOST);
        printf("Matriz interna de cada processo: %dx%d\n", linhas_total, colunas_total);
        imprimir_mapa_processos();
    }

    // loop principal (executa ate numero de geracoes definidas)
    for (int geracao = 0; geracao <= GERACOES; geracao++){
        // contagem de quantas celulas vivas existem na matriz real do processo
        int pop_local = contar_populacao_local(matriz, n_local, colunas_total);
        int pop_total = 0;

        // soma todas as populacoes locais (valor enviado, valor que vai receber (rank 0), quantidade de elementos, tipo do dado, operacao, processo root, comunicador)
        MPI_Reduce(&pop_local, &pop_total, 1, MPI_INT, MPI_SUM, 0, cart_comm);

        // junta os blocos de todos os processos no processo 0 e imprime a matriz global
        juntar_e_imprimir_global(matriz, n_local, colunas_total, rank, cart_comm, geracao, pop_total);

        // nao calcula a proxima geracao na ultima iteracao
        if (geracao < GERACOES){
            // atualiza as ghost cells de cada processo com os vizinhos
            trocar_bordas_e_cantos(matriz, n_local, colunas_total, cart_comm, cima, baixo, esquerda, direita, cima_esquerda, cima_direita, baixo_esquerda, baixo_direita);

            // cada processo aplica as regras do jogo e atualiza a matriz local
            calcular_proxima_geracao(matriz, proxima, n_local, colunas_total);
        }
    }

    // liberacao de memória e encerramento do MPI
    free(matriz);
    free(proxima);

    MPI_Finalize();
    return 0;
}