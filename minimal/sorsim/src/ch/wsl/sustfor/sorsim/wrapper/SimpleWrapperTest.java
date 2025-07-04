/*******************************************************************************
 * Copyright 2024 Stefan Holm
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/
package ch.wsl.sustfor.sorsim.wrapper;

/**
 * 
 * @author Stefan Holm
 *
 */
public class SimpleWrapperTest {
	
	public static void main(String[] args) {
		SimpleWrapper sorsim = new SimpleWrapper();

		sorsim.setFileTreeList("data/Baumliste.csv");
		sorsim.setFileAssortmentSpecifications("data/SortimentsVorgabenListe.csv");
		sorsim.setStemFormFunction_category(2);
		sorsim.setBolePercentage_pct(70);
		sorsim.setCombinationOfLengthClasses_category(5);
		
		String[][] result = sorsim.makeAssortments();
		
		for(String[] line : result) {
			for (String field : line) {
				System.out.print(field + "\t");
			}
			System.out.println();
		}

		System.out.println(sorsim.getNumberOfTrees());
		System.out.println(sorsim.getNumberOfAssortmentSpecifications());
	}
}
