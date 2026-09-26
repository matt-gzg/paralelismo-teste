#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N_GLOBAL 40
#define PROCESSOS_LADO 4
#define NUM_PROCESSOS 16
#define GHOST 5
#define GERACOES 20
#define VIVA 1
#define MORTA 0
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

void imprimir_matriz_global(int *matriz_global, int geracao, int populacao){
    printf("\n- GERACAO %d | POPULACAO TOTAL: %d -\n\n", geracao, populacao);

    for (int i = 0; i < N_GLOBAL; i++){
        for (int j = 0; j < N_GLOBAL; j++)
            printf("%c", matriz_global[i * N_GLOBAL + j] ? 'O' : '.');

        printf("\n");
    }
}

/*
    As ghost cells são exibidas como 'G'.
    As células reais aparecem como 'O' ou '.'.
*/
void imprimir_processo_com_ghosts(int *matriz, int linhas_total, int colunas_total,
                                  int rank, int proc_linha, int proc_coluna){
    printf("\n- PROCESSO %d | posicao (%d,%d) -\n", rank, proc_linha, proc_coluna);
    printf("G = ghost cell | O = viva | . = morta\n\n");

    for (int i = 0; i < linhas_total; i++){
        for (int j = 0; j < colunas_total; j++){
            int eh_ghost =
                i < GHOST ||
                i >= linhas_total - GHOST ||
                j < GHOST ||
                j >= colunas_total - GHOST;

            if (eh_ghost)
                printf("G");
            else
                printf("%c", matriz[i * colunas_total + j] ? 'O' : '.');
        }
        printf("\n");
    }
}

/*
    Conta apenas as células vivas da região real do processo.
    Ghost cells são ignoradas por serem cópias temporárias das bordas vizinhas.
*/
int contar_populacao_local(int *matriz, int n_local, int colunas_total){
    int total = 0;

    for (int i = GHOST; i < GHOST + n_local; i++)
        for (int j = GHOST; j < GHOST + n_local; j++)
            total += matriz[i * colunas_total + j];

    return total;
}

/*
    Reconstrói e imprime a matriz global a partir dos blocos locais de cada processo.

    Cada processo copia sua região real (sem ghost cells) para um vetor auxiliar.
    MPI_Gather coleta todos esses vetores no processo 0, que então os posiciona
    na matriz global de acordo com as coordenadas cartesianas de cada processo.
*/
void juntar_e_imprimir_global(int *matriz, int n_local, int colunas_total,
                              int rank, MPI_Comm cart_comm,
                              int geracao, int populacao_total){
    int *bloco_local = malloc(n_local * n_local * sizeof(int));

    for (int i = 0; i < n_local; i++)
        for (int j = 0; j < n_local; j++)
            bloco_local[i * n_local + j] =
                matriz[(i + GHOST) * colunas_total + (j + GHOST)];

    int *todos_blocos = NULL;

    if (rank == 0)
        todos_blocos = malloc(N_GLOBAL * N_GLOBAL * sizeof(int));

    MPI_Gather(
        bloco_local,
        n_local * n_local,
        MPI_INT,
        todos_blocos,
        n_local * n_local,
        MPI_INT,
        0,
        cart_comm
    );

    if (rank == 0){
        int matriz_global[N_GLOBAL * N_GLOBAL];

        for (int r = 0; r < NUM_PROCESSOS; r++){
            int coord[2];
            MPI_Cart_coords(cart_comm, r, 2, coord);

            int inicio_linha = coord[0] * n_local;
            int inicio_coluna = coord[1] * n_local;

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

/*
    Aplica as regras do Jogo da Vida sobre a região real de cada processo:
      - Célula viva com 2 ou 3 vizinhos vivos → sobrevive
      - Célula viva com menos de 2 vizinhos    → morre por solidão
      - Célula viva com mais de 3 vizinhos     → morre por superpopulação
      - Célula morta com exatamente 3 vizinhos → nasce

    O resultado é calculado em 'proxima' e depois copiado de volta para 'matriz'.
*/
void calcular_proxima_geracao(int *matriz, int *proxima, int n_local, int colunas_total){
    for (int i = GHOST; i < GHOST + n_local; i++){
        for (int j = GHOST; j < GHOST + n_local; j++){
            int vizinhos = 0;

            for (int dl = -1; dl <= 1; dl++){
                for (int dc = -1; dc <= 1; dc++){
                    if (dl == 0 && dc == 0)
                        continue;

                    vizinhos += matriz[(i + dl) * colunas_total + (j + dc)];
                }
            }

            if (matriz[i * colunas_total + j] == VIVA)
                proxima[i * colunas_total + j] =
                    (vizinhos == 2 || vizinhos == 3) ? VIVA : MORTA;
            else
                proxima[i * colunas_total + j] =
                    (vizinhos == 3) ? VIVA : MORTA;
        }
    }

    for (int i = GHOST; i < GHOST + n_local; i++)
        for (int j = GHOST; j < GHOST + n_local; j++)
            matriz[i * colunas_total + j] = proxima[i * colunas_total + j];
}

/*
    Sincroniza as ghost cells de cada processo com as bordas reais dos vizinhos.

    São trocados três tipos de regiões, cada um com um tipo MPI derivado próprio:
      - tipo_linhas : bloco de GHOST linhas × n_local colunas  (bordas superior/inferior)
      - tipo_colunas: bloco de n_local linhas × GHOST colunas  (bordas esquerda/direita)
      - tipo_canto  : bloco de GHOST × GHOST                   (quatro cantos diagonais)

    MPI_Sendrecv é usado em todas as trocas para enviar e receber simultaneamente,
    evitando deadlocks. Vizinhos inexistentes são representados por MPI_PROC_NULL,
    e as operações com esse valor são ignoradas automaticamente pelo MPI.
*/
void trocar_bordas_e_cantos(int *matriz, int n_local, int colunas_total, MPI_Comm cart_comm, int cima, int baixo, int esquerda, int direita, int cima_esquerda, int cima_direita, int baixo_esquerda, int baixo_direita){
    MPI_Datatype tipo_linhas;
    MPI_Datatype tipo_colunas;
    MPI_Datatype tipo_canto;

    MPI_Type_vector(GHOST, n_local, colunas_total, MPI_INT, &tipo_linhas);
    MPI_Type_commit(&tipo_linhas);

    MPI_Type_vector(n_local, GHOST, colunas_total, MPI_INT, &tipo_colunas);
    MPI_Type_commit(&tipo_colunas);

    MPI_Type_vector(GHOST, GHOST, colunas_total, MPI_INT, &tipo_canto);
    MPI_Type_commit(&tipo_canto);

    /* --- Troca de linhas (superior e inferior) --- */
    MPI_Sendrecv(&matriz[IDX(GHOST, GHOST)], 1, tipo_linhas, cima, 0,
                 &matriz[IDX(GHOST + n_local, GHOST)], 1, tipo_linhas, baixo, 0,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&matriz[IDX(GHOST + n_local - GHOST, GHOST)], 1, tipo_linhas, baixo, 1,
                 &matriz[IDX(0, GHOST)], 1, tipo_linhas, cima, 1,
                 cart_comm, MPI_STATUS_IGNORE);

    /* --- Troca de colunas (esquerda e direita) --- */
    MPI_Sendrecv(&matriz[IDX(GHOST, GHOST)], 1, tipo_colunas, esquerda, 2,
                 &matriz[IDX(GHOST, GHOST + n_local)], 1, tipo_colunas, direita, 2,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&matriz[IDX(GHOST, GHOST + n_local - GHOST)], 1, tipo_colunas, direita, 3,
                 &matriz[IDX(GHOST, 0)], 1, tipo_colunas, esquerda, 3,
                 cart_comm, MPI_STATUS_IGNORE);

    /* --- Troca dos quatro cantos diagonais --- */
    MPI_Sendrecv(&matriz[IDX(GHOST, GHOST)], 1, tipo_canto, cima_esquerda, 4,
                 &matriz[IDX(GHOST + n_local, GHOST + n_local)], 1, tipo_canto, baixo_direita, 4,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&matriz[IDX(GHOST, GHOST + n_local - GHOST)], 1, tipo_canto, cima_direita, 5,
                 &matriz[IDX(GHOST + n_local, 0)], 1, tipo_canto, baixo_esquerda, 5,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&matriz[IDX(GHOST + n_local - GHOST, GHOST)], 1, tipo_canto, baixo_esquerda, 6,
                 &matriz[IDX(0, GHOST + n_local)], 1, tipo_canto, cima_direita, 6,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&matriz[IDX(GHOST + n_local - GHOST, GHOST + n_local - GHOST)], 1, tipo_canto, baixo_direita, 7,
                 &matriz[IDX(0, 0)], 1, tipo_canto, cima_esquerda, 7,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Type_free(&tipo_linhas);
    MPI_Type_free(&tipo_colunas);
    MPI_Type_free(&tipo_canto);
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int rank, total_processos;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processos);

    if (total_processos != NUM_PROCESSOS){
        if (rank == 0)
            printf("Execute com: mpirun -np 16 ./jogo_mpi\n");

        MPI_Finalize();
        return 1;
    }

    /*
        Cria um comunicador cartesiano 4×4 sem periodicidade nas bordas.
        Isso permite usar MPI_Cart_shift e MPI_Cart_coords para descobrir
        vizinhos e coordenadas de cada processo na grade.
    */
    int dimensoes[2] = {4, 4};
    int periodico[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensoes, periodico, 0, &cart_comm);

    int coordenadas[2];
    MPI_Cart_coords(cart_comm, rank, 2, coordenadas);

    int proc_linha = coordenadas[0];
    int proc_coluna = coordenadas[1];

    /*
        Obtém os vizinhos diretos (cima/baixo e esquerda/direita) via MPI_Cart_shift.
        Os vizinhos diagonais são calculados manualmente a partir das coordenadas,
        e recebem MPI_PROC_NULL quando estão fora dos limites da grade.
    */
    int cima, baixo, esquerda, direita;
    MPI_Cart_shift(cart_comm, 0, 1, &cima, &baixo);
    MPI_Cart_shift(cart_comm, 1, 1, &esquerda, &direita);

    int cima_esquerda  = MPI_PROC_NULL;
    int cima_direita   = MPI_PROC_NULL;
    int baixo_esquerda = MPI_PROC_NULL;
    int baixo_direita  = MPI_PROC_NULL;

    int coord_temp[2];

    coord_temp[0] = proc_linha - 1; coord_temp[1] = proc_coluna - 1;
    if (coord_temp[0] >= 0 && coord_temp[1] >= 0)
        MPI_Cart_rank(cart_comm, coord_temp, &cima_esquerda);

    coord_temp[0] = proc_linha - 1; coord_temp[1] = proc_coluna + 1;
    if (coord_temp[0] >= 0 && coord_temp[1] < PROCESSOS_LADO)
        MPI_Cart_rank(cart_comm, coord_temp, &cima_direita);

    coord_temp[0] = proc_linha + 1; coord_temp[1] = proc_coluna - 1;
    if (coord_temp[0] < PROCESSOS_LADO && coord_temp[1] >= 0)
        MPI_Cart_rank(cart_comm, coord_temp, &baixo_esquerda);

    coord_temp[0] = proc_linha + 1; coord_temp[1] = proc_coluna + 1;
    if (coord_temp[0] < PROCESSOS_LADO && coord_temp[1] < PROCESSOS_LADO)
        MPI_Cart_rank(cart_comm, coord_temp, &baixo_direita);

    /*
        Cada processo opera sobre um bloco real de n_local×n_local células.
        A matriz local inclui ainda uma borda de GHOST células em cada lado,
        resultando em (n_local + 2*GHOST)² elementos alocados por processo.
        calloc garante que todas as células comecem mortas (valor 0).
    */
    int n_local = N_GLOBAL / PROCESSOS_LADO;
    int linhas_total   = n_local + 2 * GHOST;
    int colunas_total  = n_local + 2 * GHOST;

    int *matriz  = calloc(linhas_total * colunas_total, sizeof(int));
    int *proxima = calloc(linhas_total * colunas_total, sizeof(int));

    /*
        Padrão inicial: cruz 3×3 posicionada próxima ao centro da matriz global.
        Cada processo verifica quais células do padrão pertencem ao seu bloco real
        e as marca como vivas, convertendo coordenadas globais para locais.
    */
    int deslocamento_linha   = 19;
    int deslocamento_coluna  = 19;

    int padrao[][2] = {
                {0, 1},
        {1, 0}, {1, 1}, {1, 2},
                {2, 1}
    };

    int qtd_padrao = sizeof(padrao) / sizeof(padrao[0]);

    for (int k = 0; k < qtd_padrao; k++){
        int linha_global   = padrao[k][0] + deslocamento_linha;
        int coluna_global  = padrao[k][1] + deslocamento_coluna;

        int inicio_linha   = proc_linha   * n_local;
        int inicio_coluna  = proc_coluna  * n_local;
        int fim_linha      = inicio_linha  + n_local - 1;
        int fim_coluna     = inicio_coluna + n_local - 1;

        if (linha_global >= inicio_linha && linha_global <= fim_linha &&
            coluna_global >= inicio_coluna && coluna_global <= fim_coluna)
        {
            int linha_local   = linha_global  - inicio_linha  + GHOST;
            int coluna_local  = coluna_global - inicio_coluna + GHOST;
            matriz[IDX(linha_local, coluna_local)] = VIVA;
        }
    }

    if (rank == 0){
        printf("Matriz global: %dx%d\n", N_GLOBAL, N_GLOBAL);
        printf("Processos: %d\n", NUM_PROCESSOS);
        printf("Divisao: 4x4 processos\n");
        printf("Cada processo: %dx%d celulas reais\n", n_local, n_local);
        printf("Ghost cells: largura %d ao redor de cada bloco\n", GHOST);
        printf("Matriz interna de cada processo: %dx%d\n", linhas_total, colunas_total);
        imprimir_mapa_processos();
    }

    /*
        Loop principal da simulação (geração 0 = estado inicial).

        A cada iteração:
          1. Conta a população local e reduz a soma global no processo 0;
          2. Reconstrói e imprime a matriz global;
          3. Troca ghost cells com os vizinhos;
          4. Calcula a próxima geração.

        A troca e o cálculo são omitidos na última geração.
    */
    for (int geracao = 0; geracao <= GERACOES; geracao++){
        int pop_local = contar_populacao_local(matriz, n_local, colunas_total);
        int pop_total = 0;
        MPI_Reduce(&pop_local, &pop_total, 1, MPI_INT, MPI_SUM, 0, cart_comm);

        juntar_e_imprimir_global(matriz, n_local, colunas_total,
                                 rank, cart_comm, geracao, pop_total);

        if (geracao < GERACOES){
            trocar_bordas_e_cantos(matriz, n_local, colunas_total,
                                   cart_comm,
                                   cima, baixo, esquerda, direita,
                                   cima_esquerda, cima_direita,
                                   baixo_esquerda, baixo_direita);

            calcular_proxima_geracao(matriz, proxima, n_local, colunas_total);
        }
    }

    free(matriz);
    free(proxima);

    MPI_Finalize();
    return 0;
}