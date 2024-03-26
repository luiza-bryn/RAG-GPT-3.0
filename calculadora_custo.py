import tiktoken

dolar = 4.97
modelo = "gpt-4-0125-preview"
prompt_sistema = """
Você é um sistema que auxilia na leitura e na classificação de Startups perante a um edital focado em selecionar Startups para algum próposito. A entrada é todo o conteúdo do website de algum edital voltado a Startups.  Classifique, a partir do conteúdo de entrada, o nível de maturidade o qual a startup deverá apresentar para a inscrição no edital.

Sua saída é no máximo DUAS classificações das mencionadas abaixo. Caso não tenha informações suficientes no edital para a classificação, sua saída será "INDEFINIDO":

##### NÍVEIS DE STARTUP 
    - IDEACAO: Empreendedor potencial trabalhando em uma ideia de negócio. É o estágio inicial.
    - VALIDACAO: Uma startup iterando em direção ao encaixe entre Problema e Solução, validando MVPs enquanto busca clientes pagantes.
    - TRACAO: Empresa com um portfólio estável de clientes pagantes, consistentemente melhorando sua operação em direção à previsibilidade de receita e o encaixe entre Produto e Mercado.
    - CRESCIMENTO: Empresa em crescimento consistente após o encaixe entre Produto e Mercado, otimizando a gestão para acelerar vendas e escalar o negócio.
    - ESCALA: Scale-up crescendo cem por cento ou mais ao ano, buscando diversificação e consolidação.

##### EXEMPLO DE ENTRADA
[conteúdo web do edital]

##### EXEMPLO DE SAÍDA

IDEACAO, VALIDACAO
"""
prompt = """
Ambev Startups
Este é o portão de entrada para empreendedoras e empreendedores que querem se relacionar com a nossa companhia. Nossa missão aqui é viabilizar oportunidades de inovação por meio de interação com startups.

Bem vindo ao Ambev Startups!

CADASTRO GERAL DE STARTUPS

Esse é o nosso portão de entrada, independente de verticais ou desafios.

ATENÇÃO: SE VOCÊ BUSCA INFORMAÇÕES SOBRE A ACELERADORA 100+ DE SUSTENTABILIDADE, ACESSE: https://aceleradora.ambev.com.br/

CHAMADAS DE INOVAÇÃO ABERTA

Boa vindas à nossa página, aqui sempre atualizamos nossos desafios. 

F.A.Q

Startups de qualquer cidade serão aceitas? Sim. Podem se candidatar aos desafios startups de qualquer localidade.

Toda startup inscrita irá apresentar sua solução? Não necessariamente. Os envios de pitch decks passam por um processo de pré-seleção, realizado pelo time de inovação e tecnologias da Ambev. Apenas as startups pré-selecionadas apresentarão suas soluções, podendo ser de forma presencial ou remota.

Como funciona a metodologia? Após a aprovação da sua startup, é feito um alinhamento para definir o tipo de relacionamento de inovação ideal. Temos 10 tipos de relacionamento mapeados: Validação de Conceito (PoC), Desenvolvimento e Validação de MVP, Aprimoramento da Solução, Co-desenvolvimento, Parceria de Tecnologia, Parceria Comercial, OEM ou White Label, Venda B2B Direta, Mentoria e Corporate Venture Capital. Para cada tipo de relacionamento, temos um método de acompanhamento dos projetos, garantindo, assim, o andamento do projeto.

O programa envolve investimentos e/ou o equity da startup? Não. O principal objetivo do programa é realizar a conexão da startup com Ambev. Realizamos essa conexão sem exigir nenhum equity da sua empresa e, a partir da conexão realizada, negociaremos o formato de relacionamento.

Preciso pagar para me inscrever? Não. A inscrição no programa é gratuita.

Ainda não tenho nada validado, possuo apenas uma ideia, posso me inscrever?  Pode. No programa temos startups dos mais diversos níveis de maturidade, mas startups com um nível mínimo de desenvolvimento costumam ter mais chances de sucesso.

Se minha solução não for selecionada, eu posso inscrevê-la novamente? Sim, depois de 3 meses. Analisaremos a evolução nesse período e leremos sua proposta novamente.

Como sei se minha startup foi selecionada para a próxima fase? Em até 90 dias após o envio, daremos um feedback através do e-mail usando no momento da inscrição
"""
# INFORMAÇÃO RETIRADA DO SITE DA OPENAI [29/02/2014]
"""
gpt-4-0125-preview	$0.01 / 1K tokens	$0.03 / 1K tokens
gpt-4-1106-preview	$0.01 / 1K tokens	$0.03 / 1K tokens
gpt-4-1106-vision-preview	$0.01 / 1K tokens	$0.03 / 1K tokenss
gpt-4	$0.03 / 1K tokens	$0.06 / 1K tokens
gpt-4-32k	$0.06 / 1K tokens	$0.12 / 1K tokens
gpt-3.5-turbo-0125	$0.0005 / 1K tokens	$0.0015 / 1K tokens
gpt-3.5-turbo-instruct	$0.0015 / 1K tokens	$0.0020 / 1K tokens

"""


dict_modelo = {"gpt-4-0125-preview": 0.01, "gpt-4": 0.03, "gpt-4-32k": 0.06, "gpt-3.5-turbo-0125": 0.0005, "gpt-3.5-turbo-instruct": 0.0015, "gpt-3.5-turbo": 0.0015}


codificador = tiktoken.encoding_for_model(modelo)
lista_tokens_entrada = codificador.encode(prompt+prompt_sistema)
lista_tokens_saida = codificador.encode("IDEACAO, VALIDACAO")
nro_tokens_entrada = len(lista_tokens_entrada)
nro_tokens_saida = len(lista_tokens_saida)
nro_tokens_total = nro_tokens_saida + nro_tokens_entrada
valor = dict_modelo[modelo]
custo = float(nro_tokens_total/1000 * valor)

print(f"Com um prompt de {nro_tokens_total} tokens o custo será de {custo} dolares ou {custo*dolar} reais")

