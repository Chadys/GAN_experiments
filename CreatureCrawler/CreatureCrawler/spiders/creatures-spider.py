import scrapy


class PokemonSpider(scrapy.Spider):
    name = "pokemon-images-spider"
    start_urls = ["https://www.pokemon.com/us/pokedex/bulbasaur"]

    def parse(self, response):
        img_url = response.css('img.active::attr(src)').extract_first()
        yield {'image_urls': [img_url]}

        next_page = response.css('a.next::attr(href)').extract_first()
        if next_page not in self.start_urls:
            yield response.follow(next_page, callback=self.parse)


class YugiohSpider(scrapy.Spider):
    name = "yugioh-images-spider"
    start_urls = ["https://www.yugioh.com/cards?archetype_id=1"]

    def parse(self, response):
        for card in response.css("ul.cards-list li"):
            next_page = card.css("a::attr(href)").extract_first()
            yield response.follow(next_page, callback=self.parse_card)

        next_page = response.css('div.pagination a.btn-next::attr(href)').extract_first()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)

    @staticmethod
    def parse_card(response):
        img_url = response.css('section.card-promo div.holder div.promo-container img::attr(src)').extract_first()
        yield {'image_urls': [img_url]}


class MagicSpider(scrapy.Spider):
    name = "magic-images-spider"
    main_url = "http://www.magiccorporation.com/mc.php?rub=cartes&op=search&search=2&bool_mana=0&mode=list&lang_vf=1&lang_vo=1&nom=&nom_type=&texte=&cout_mana_egal=&cout_mana_maxi=&cout_mana_mini=&type_vf%5B0%5D=Cr%E9ature&force_egal=&force_maxi=&force_mini=&endurance_egal=&endurance_maxi=&endurance_mini=&bool_type=1&bool_creature=1&bool_capacite=0&bool_illustrateur=1&limit="
    # current_entry = 0
    # step_entry = 500
    # max_entry = 1500
    start_urls = [main_url+"0"]

    def parse(self, response):
        for card in response.css("table.editions tbody tr"):
            next_page = card.css("a::attr(href)").extract_first()
            yield response.follow(response.urljoin(next_page), callback=self.parse_card)

        # self.current_entry += self.step_entry
        # if self.current_entry > self.max_entry:
        #     return
        # yield response.follow(self.main_url+str(self.current_entry), callback=self.parse)
        next_page = response.css('div.html_link_search') \
            .xpath("//a[contains(@title, 'Page Suivante')][contains(., '>')]/@href").extract_first()
        if next_page is not None:
            yield response.follow(response.urljoin(next_page), callback=self.parse)


    @staticmethod
    def parse_card(response):
        # exclude flippable cards
        if response.css('div.html_div div.block div div.block_content::text')\
                .re('}———————{ Flip }———————{'):
            return
        img_url = response.css('div.html_div div img::attr(src)').extract_first()
        yield {'image_urls': [response.urljoin(img_url)]}



